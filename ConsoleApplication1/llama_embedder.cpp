#include "llama_embedder.h"
#include "llama.h"
#include <windows.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace
{
    typedef void (*fp_llama_backend_init)(void);
    typedef void (*fp_llama_backend_free)(void);
    typedef struct llama_model_params(*fp_llama_model_default_params)(void);
    typedef struct llama_model * (*fp_llama_model_load_from_file)(const char *, struct llama_model_params);
    typedef void (*fp_llama_model_free)(struct llama_model *);
    typedef bool (*fp_llama_model_has_encoder) (const struct llama_model *);
    typedef int32_t (*fp_llama_model_n_embd)(const struct llama_model *);
    
    typedef struct llama_context_params (*fp_llama_context_default_params)(void); 
    typedef struct llama_context *(*fp_llama_init_from_model)(struct llama_model *, struct llama_context_params);
    typedef void  (*fp_llama_free)(struct llama_context *);
    typedef void (*fp_llama_set_n_threads)(struct llama_context *, int32_t, int32_t);
    
    typedef void (*fp_llama_set_embeddings)(struct llama_context *, bool);
    typedef const struct llama_vocab * (*fp_llama_model_get_vocab)(const struct llama_model *);
    typedef int32_t (*fp_llama_vocab_n_tokens)(const struct llama_vocab *);
    typedef int32_t (*fp_llama_tokenize)(const struct llama_vocab *, const char *, int32_t, llama_token *, int32_t, bool, bool);
    typedef struct llama_batch (*fp_llama_batch_get_one)(llama_token *, int32_t);
    typedef int32_t (*fp_llama_encode)(struct llama_context *, struct llama_batch);
    typedef float *(*fp_llama_get_embeddings_seq)(struct llama_context *, llama_seq_id);
    typedef void (*fp_llama_log_set)(ggml_log_callback, void *);
    // ggmal.dll
    typedef void (*fp_ggml_backend_load_all_from_path)(const char *);
    
    //статические указатели
    static fp_llama_backend_init            fn_backend_init = nullptr;
    static fp_llama_backend_free            fn_backend_free = nullptr;
    static fp_llama_model_default_params            fn_model_default_params = nullptr;
    static fp_llama_model_load_from_file         fn_model_load_from_file = nullptr;
    static fp_llama_model_free            fn_model_free = nullptr;
    static fp_llama_model_has_encoder     fn_model_has_encoder = nullptr;
    static fp_llama_model_n_embd         fn_model_n_embd = nullptr;
    static fp_llama_context_default_params    fn_context_default_params = nullptr;
    static fp_llama_init_from_model         fn_init_from_model = nullptr;
    static fp_llama_free                    fn_free = nullptr;
    static fp_llama_set_n_threads            fn_set_n_threads = nullptr;
    static fp_llama_set_embeddings          fn_set_embeddings = nullptr;
    static fp_llama_model_get_vocab         fn_model_get_vocab = nullptr;
    static fp_llama_vocab_n_tokens          fn_vocab_n_tokens = nullptr;
    static fp_llama_tokenize                fn_tokenize = nullptr;
    static fp_llama_batch_get_one          fn_batch_get_one = nullptr;
    static fp_llama_encode                 fn_encode = nullptr;
    static fp_llama_get_embeddings_seq      fn_get_embeddings_seq = nullptr;
    static fp_llama_log_set                 fn_log_set = nullptr;
    
    static fp_ggml_backend_load_all_from_path           fn_backend_load_all_from_path = nullptr;
    static bool g_resolved = false; // один раз на процесс
    
    //вспомогательные
    
    template <typename T>
    static bool resolve(HMODULE mod, const char *name, T &out)
    {
        FARPROC p = GetProcAddress(mod, name);
        if (!p) return false;
        out = reinterpret_cast<T>(p);
        return true;
    }
    
    static std::string g_last_error;
    static void set_error(const std::string &msg)
    {
        g_last_error = msg;
    }
    // UTF-8 
    static wchar_t *utf8_to_wide(const char *s)
    {
        int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
        if (n <= 0) return nullptr;
        wchar_t *w = new wchar_t[(size_t)n];
        MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n);
        return w;
    }
    // загрузить dll (filesystem::path → wstring, работает с русскими буквами)
    static HMODULE load_dll(const std::filesystem::path& p)
    {
        std::wstring ws = p.wstring();
        return LoadLibraryExW(ws.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    
}

const char* LlamaCppEmbedder::last_error() {
    return g_last_error.c_str();
}

LlamaCppEmbedder::LlamaCppEmbedder() = default;

LlamaCppEmbedder::~LlamaCppEmbedder()
{
    if (ctx_)   { fn_free(static_cast<llama_context*>(ctx_)); ctx_ = nullptr; }
    if (model_) { fn_model_free(static_cast<llama_model *>(model_)); model_ = nullptr; }
    if (h_llama_) { FreeLibrary(static_cast<HMODULE>(h_llama_)); h_llama_ = nullptr; }
    if (h_ggml_) { FreeLibrary(static_cast<HMODULE>(h_ggml_)); h_ggml_ = nullptr; }
    loaded_ = false;
}
bool LlamaCppEmbedder::init(const char *llama_dll_dir, const char *model_path, int n_threads)
{
    if (loaded_) return true;
    //загрузить llama.dll
    std::filesystem::path llama_dir(llama_dll_dir);
    std::filesystem::path llama_dll = llama_dir / "llama.dll";
    h_llama_ = load_dll(llama_dll);
    if (!h_llama_)
    {
        set_error("init: не могу загрузить llama.dll: " + llama_dll.string());
        return false;
    }
    // загрузить ggml.dll
    std::filesystem::path ggml_dll = llama_dir / "ggml.dll";
    h_ggml_ = load_dll(ggml_dll);
    if (!h_ggml_)
    {
        set_error("init: не могу загрузить ggml.dll: " + ggml_dll.string());
        return false;
    }
    // разделить все символы 
    HMODULE hl = static_cast<HMODULE>(h_llama_);
    HMODULE hg = static_cast<HMODULE>(h_ggml_);
#define R(mod, fn, sym) \
if (!resolve(mod, sym, fn)) { \
set_error("init: нет символа " sym); \
return false; \
}

    
    R(hl, fn_backend_init, "llama_backend_init");
    R(hl, fn_backend_free, "llama_backend_free");
    R(hl, fn_model_default_params, "llama_model_default_params");
    R(hl, fn_model_load_from_file, "llama_model_load_from_file");
    R(hl, fn_model_free, "llama_model_free");
    R(hl, fn_model_has_encoder, "llama_model_has_encoder");
    R(hl, fn_model_n_embd, "llama_model_n_embd");
    R(hl, fn_context_default_params, "llama_context_default_params");
    R(hl, fn_init_from_model, "llama_init_from_model");
    R(hl, fn_free, "llama_free");
    R(hl, fn_set_n_threads, "llama_set_n_threads");
    R(hl, fn_set_embeddings, "llama_set_embeddings");
    R(hl, fn_model_get_vocab, "llama_model_get_vocab");
    R(hl, fn_vocab_n_tokens, "llama_vocab_n_tokens");
    R(hl, fn_tokenize, "llama_tokenize");
    R(hl, fn_batch_get_one, "llama_batch_get_one");
    R(hl, fn_encode, "llama_encode");
    R(hl, fn_get_embeddings_seq, "llama_get_embeddings_seq");
    R(hl, fn_log_set, "llama_log_set");
    
    R(hg, fn_backend_load_all_from_path, "ggml_backend_load_all_from_path");
#undef R
    
    g_resolved = true;

    fn_backend_init();
    fn_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    // llama_backend_init() уже вызывает ggml_backend_load_all_from_path внутри —
    // повторный вызов даёт stack buffer overrun (0xC0000409) на b10063.

    //загрузить модель
    llama_model_params mparams = fn_model_default_params();
    llama_model *model = fn_model_load_from_file(model_path, mparams);
    if (!model)
    {
        set_error("init: llama_model_load_from_file: " + std::string(model_path));
        return false;
    }
    // b10063: llama_model_has_encoder может вернуть false для bert-моделей,
    // хотя модель реально энкодерная. Пропускаем проверку.
    // создать контекст
    llama_context_params cparams = fn_context_default_params();
    cparams.n_ctx   = 2048; // BGE-M3 поддерживает 8192, берём 2048
    cparams.n_batch = 2048;
    cparams.n_ubatch = 2048;
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    // pooling
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN; 
    
    llama_context *ctx = fn_init_from_model(model, cparams);
    if (!ctx)
    {
        set_error("init: llama_init_from_model failed"); 
        fn_model_free(model);
        return false;
    }
    fn_set_n_threads(ctx, n_threads, n_threads);
    fn_set_embeddings(ctx, true); // включить вывод эмбеддингов
    
    // сохранить состояние
    model_ = model;
    ctx_ = ctx;
    vocab_ = fn_model_get_vocab(model);
    dim_ = (int)fn_model_n_embd(model);
    loaded_ = true;
    
    return true;
}

std::vector<float>LlamaCppEmbedder::embed(std::string_view text) const
{
    if (!loaded_ || !ctx_ || !vocab_)
        return{};
    
    auto *ctx = static_cast<llama_context *>(ctx_);
    auto *vocab = static_cast<const llama_vocab *>(vocab_);
    
    //токенизация - явная длина, string_view не null-terminated
    const int MAX_TOKENS = 2046; // BGE-M3 context = 2048, минус BOS/EOS
    std::vector<llama_token> tokens((size_t)MAX_TOKENS + 2);
    int n_tokens = fn_tokenize(vocab, text.data(), (int32_t)text.size(),
                               tokens.data(), MAX_TOKENS, true, true);
    if (n_tokens < 0)
        return {}; // слишком длинный текст
    if (n_tokens > MAX_TOKENS) n_tokens = MAX_TOKENS;
    
    if (n_tokens == 0)
    {
        return std::vector<float>((size_t)dim_, 0.0f); // пустой ввод -> нулевой вектор
    }
    //батч из одного предложения seq_id = 0
    llama_batch batch = fn_batch_get_one(tokens.data(), n_tokens);
    //энкодер, не трогает kv кэш каждый вызов независим
    if (fn_encode(ctx, batch) != 0)
        return {};
    
    //pooled embedding для seq_id=0
    float *emb = fn_get_embeddings_seq(ctx, 0);
    if (!emb)
        return {};
    
    //копировать + L2 нормировать
    std::vector<float> result((size_t)dim_);
    float n2 = 0.0f;
    for (int i = 0; i < dim_; ++i)
    {
        result[i] = emb[i];
        n2 += emb[i] * emb[i];
    }
    if (n2 > 0.0f)
    {
        float inv = 1.0f / std::sqrt(n2);
        for (float &v : result) v *= inv;
    }
    return result;
}

