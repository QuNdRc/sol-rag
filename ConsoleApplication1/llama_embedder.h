#pragma once
#include "rag_core.h"
#include <string>
#include <vector>

class LlamaCppEmbedder : public Embedder
{
public:
    LlamaCppEmbedder();
    ~LlamaCppEmbedder();

    LlamaCppEmbedder(const LlamaCppEmbedder &) = delete;
    LlamaCppEmbedder &operator=(const LlamaCppEmbedder &) = delete;

    /// Загрузить llama.dll/ggml.dll, backend, модель.
    /// @param llama_dll_dir  папка с llama.dll, ggml.dll, ggml-cpu-*.dll
    /// @param model_path     путь к .gguf
    /// @param n_threads      0 = hardware_concurrency()
    bool init(const char* llama_dll_dir, const char* model_path, int n_threads = 0);

    bool is_loaded() const { return loaded_; }

    /// Текст последней ошибки (после неудачного init).
    static const char* last_error();

    // Размерность эмбеддинга (384 для MiniLM).
    int dim() const { return dim_; }

    // Эмбеддинг текста (L2-нормированный, размерность dim()).
    // Вызывает llama_encode — не использует KV-кеш, каждый вызов независим.
    std::vector<float> embed(std::string_view text) const override;

private:
    void* h_llama_ = nullptr;   // HMODULE llama.dll
    void* h_ggml_  = nullptr;   // HMODULE ggml.dll
    void* model_   = nullptr;   // llama_model*
    void* ctx_     = nullptr;   // llama_context*
    const void* vocab_ = nullptr; // const llama_vocab*
    int  dim_    = 0;
    bool loaded_ = false;
};
