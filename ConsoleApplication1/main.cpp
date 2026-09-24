#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <clocale>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <filesystem>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // SetConsoleCP / SetConsoleOutputCP
#include <io.h>      // _isatty / _fileno
#include <conio.h>   // _getch / _kbhit
#else
#include <unistd.h>  // isatty (POSIX-fallback)
#endif
#include "glav.h"
#include "rag_core.h"
#include "embedder.h"
#include "llama_embedder.h"
#include "bm25_index.h"

// Фильтр мусорных чанков: _________________, обрывки таблиц, строки
// короче 30 символов или состоящие в основном из не-букв.
static bool is_garbage_chunk(std::string_view c)
{
    // Пустой или только пробелы/непечатные символы
    size_t first = c.find_first_not_of(" \t\r\n\v\f");
    if (first == std::string_view::npos) return true;
    c = c.substr(first);

    // Слишком короткий
    if (c.size() < 40) return true;

    // Считаем буквы и регистр (UTF-8 aware: старший бит + латиница)
    size_t letters = 0, upper = 0, digits = 0;
    for (unsigned char ch : c) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch >= 0x80) {
            ++letters;
            if (ch >= 'A' && ch <= 'Z') ++upper;
        } else if (ch >= '0' && ch <= '9') {
            ++digits;
        }
    }
    float ratio = (float)letters / (float)c.size();
    // Меньше 25% букв — мусор (таблицы, разделители)
    if (ratio < 0.25f) return true;
    // Почти одни цифры — обрывок таблицы
    if ((float)digits / (float)c.size() > 0.60f) return true;
    // Заглавных букв больше 50% от всех букв и чанк < 300 символов —
    // это метаданные: «УТВЕРЖДЕН», «ИСПОЛНИТЕЛИ», заголовки.
    if (letters > 10 && (float)upper / (float)letters > 0.50f && c.size() < 300)
        return true;
    // Начинается с цифры + пробел и длина < 250 → метаданные/оглавление:
    // «1 ИСПОЛНИТЕЛИ», «2 ВНЕСЕН», «4 УТВЕРЖДЕН», «5.1 Общие положения»
    if (c.size() < 250 && c[0] >= '0' && c[0] <= '9'
        && (c[1] == ' ' || c[1] == '.' || c[1] == '\t'))
        return true;
    // Чёрный список фраз метаданных (заголовки документов, штампы)
    // «4    УТВЕРЖДЕН приказом...» проходит другие фильтры из-за длины и
    // Unicode-пробелов — ловим по ключевым фразам.
    if (c.find("УТВЕРЖДЕН приказом") != std::string_view::npos && c.size() < 400) return true;
    if (c.find("ПОДГОТОВЛЕН к утверждению") != std::string_view::npos) return true;
    if (c.find("Сведения о своде правил") != std::string_view::npos) return true;
    if (c.find("Актуализированная редакция") != std::string_view::npos && c.size() < 300) return true;
    if (c.find("ЗАРЕГИСТРИРОВАН Федеральным") != std::string_view::npos) return true;
    if (c.find("ВНЕСЕН Техническим") != std::string_view::npos) return true;
    if (c.find("ИСПОЛНИТЕЛИ - ООО") != std::string_view::npos) return true;
    // Преамбулы документов: «1.1 Настоящий свод правил...», «1.2 ... не распространяется...»
    // Это «Область применения» — бесполезны для конкретных запросов,
    // но доминируют в выдаче из-за семантической близости к любым строительным темам.
    if (c.find("Настоящий свод правил следует соблюдать") != std::string_view::npos
        && c.size() < 2000) return true;
    if (c.find("свод правил не распространяется на") != std::string_view::npos
        && c.size() < 2000) return true;
    if (c.find("Требования к котельным, а также к связанным") != std::string_view::npos
        && c.size() < 2000) return true;
    if (c.find("строительство, реконструкция и капитальный ремонт которых осуществляются") != std::string_view::npos
        && c.size() < 500) return true;
    return false;
}

// (rrf_merge переехал в rag_core.h — теперь часть публичного ядра)

struct ParagraphChunker : Chunker
{
        std::vector<std::string> chunk(const std::string& dco) const override
        {
                std::vector<std::string> out;
                std::string cur;
                std::istringstream ss(dco);
                std::string line;
                bool any = false;
                while (std::getline(ss, line))
                {
                        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
                                line.pop_back(); // CRLF и хвостовые пробелы
                        if (line.empty())
                        {
                                if (any) { out.push_back(cur); cur.clear(); any = false; }
                        }
                        else
                        {
                                any = true;
                                if (!cur.empty()) cur += ' ';
                                cur += line;
                        }
                }
                if (any)
                        out.push_back(cur);
                return out;
        }
};

// Найти папку documents: сначала рядом с CWD, затем поднимаясь от папки exe вверх
std::filesystem::path find_docs(char* argv0)
{
        std::filesystem::path cwd = std::filesystem::current_path();
        if (std::filesystem::exists(cwd / "documents"))
                return cwd / "documents";

        std::filesystem::path start;
        if (argv0 && *argv0)
                start = std::filesystem::absolute(argv0).parent_path();
        else
                start = cwd;

        std::filesystem::path p = start;
        for (int i = 0; i < 6 && !p.empty(); ++i, p = p.parent_path())
        {
                std::filesystem::path cand = p / "documents";
                if (std::filesystem::exists(cand))
                        return cand;
        }
        return cwd / "documents";
}

// --- Работа с «живой» консолью ---
// stdin/stdout могут быть перенаправлены (пайп, панель Run в IDE, редирект).
// Тогда клавиатуры нет вообще, и getline мгновенно получит EOF,
// а консольное окно «вылупится» (пропадёт) само. _isatty() говорит правду.
static bool stdin_is_tty()
{
#ifdef _WIN32
        return _isatty(_fileno(stdin)) != 0;
#else
        return isatty(fileno(stdin)) != 0;
#endif
}

static bool stdout_is_tty()
{
#ifdef _WIN32
        return _isatty(_fileno(stdout)) != 0;
#else
        return isatty(fileno(stdout)) != 0;
#endif
}

// Держим консольное окно открытым после выхода (двойной клик в Explorer),
// чтобы оно не закрывалось мгновенно и результат можно было прочитать.
static void console_pause()
{
#ifdef _WIN32
        while (_kbhit()) _getch(); // вычистить случайно набитый ввод
        std::cout << "\n[CONSOLE] Нажми любую клавишу, чтобы закрыть окно...\n";
        _getch();
#endif
}

int main(int argc, char* argv[])
{
#ifdef _WIN32
        // Свои кодовые страницы: клавиатура и экран как UTF-8. Тогда русские
        // запросы работают в любой консоли даже без ручного `chcp 65001`.
        SetConsoleOutputCP(65001);
        SetConsoleCP(65001);
        // CRT (fopen и др.) тоже на UTF-8 — чтобы llama.cpp открывал
        // файлы с русскими буквами в путях (нужен Win10 1903+).
        setlocale(LC_ALL, ".utf8");
#endif
        
        
        // 0) Чанки из documents/ (нужны и для TF-IDF, и для семантики)
        const std::filesystem::path docs = find_docs(argc > 0 ? argv[0] : nullptr);
        std::vector<std::string> chunks;
        {
                ParagraphChunker chunker;
                if (!std::filesystem::exists(docs))
                {
                        std::cout << "[SYSTEM] Папка 'documents' не найдена\n";
                        return 1;
                }
                for (auto& entry : std::filesystem::directory_iterator(docs))
                {
                        if (entry.path().extension() != ".txt")
                                continue;
                        
                        std::ifstream ifs(entry.path(), std::ios::binary);
                        std::string text((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                        auto paras = chunker.chunk(text);
                        for (auto& p : paras)
                                if (!p.empty())
                                    chunks.push_back(p);
                }
                std::cout << "[SYSTEM] Чанков из documents/: " << chunks.size() << "\n";
        }

        // Отфильтровать мусор (______, обрывки таблиц)
        {
            std::vector<std::string> clean;
            for (auto& c : chunks)
                if (!is_garbage_chunk(c))
                    clean.push_back(std::move(c));
            size_t dropped = chunks.size() - clean.size();
            chunks = std::move(clean);
            if (dropped > 0)
                std::cout << "[SYSTEM] Отфильтровано мусора: " << dropped << " → осталось "
                          << chunks.size() << " чанков\n";
        }

        // ==== ГИБРИДНЫЙ RAG: dense (BGE-M3) + sparse (BM25) под одним фасадом ====
        std::filesystem::path exe_dir = std::filesystem::absolute(
            argc > 0 ? argv[0] : ".").parent_path();
        std::filesystem::path model_path = docs.parent_path() / "models"
            / "Bge-M3-567M-Q8_0.gguf";

        LlamaCppEmbedder sem_emb;
        if (!sem_emb.init(exe_dir.string().c_str(), model_path.string().c_str()))
        {
            std::cout << "[FATAL] LlamaCppEmbedder не загрузился: "
                      << LlamaCppEmbedder::last_error() << "\n";
            return 1;
        }
        std::cout << "[SYSTEM] Семантический эмбеддер загружен, dim="
                  << sem_emb.dim() << "\n";

        // HybridRag держит dense (RAG) и sparse (BM25) индексы в синхроне:
        // add_chunk кладёт в оба, search ищет по обоим и сливает через RRF.
        HybridRag hybrid(sem_emb, static_cast<size_t>(sem_emb.dim()));

        const std::string sem_rag_path = (docs / "rag_sem.bin").string();
        if (std::filesystem::exists(sem_rag_path))
        {
            if (!hybrid.load(sem_rag_path))
            {
                std::cout << "[SYSTEM] rag_sem.bin битый — удали и перезапусти\n";
                return 1;
            }
            std::cout << "[SYSTEM] Индекс загружен: " << hybrid.size()
                      << " чанков, dim=" << hybrid.dim()
                      << " (BM25 пересобран из текстов)\n";
        }
        else
        {
            for (size_t i = 0; i < chunks.size(); ++i)
                hybrid.add_chunk(chunks[i]);
            hybrid.save(sem_rag_path);
            std::cout << "[SYSTEM] Индекс построен: " << hybrid.size()
                      << " чанков, dim=" << hybrid.dim() << "\n";
        }

        // (демо-запросы убраны: программа сразу переходит к бенчмарку и REPL —
        //  захардкоженные вопросы про кухню/лестницу только мозолили глаза)
        constexpr size_t DIM = 384;
        constexpr size_t N = 20000; // 20к чанков на 30мб флоат
        VectorStorage big;
        big.dim = DIM;
        std::vector<float> vec(DIM);
        std::srand(7);
        for (size_t i = 0; i < N; ++i)
        {
                float n2 = 0.0f;
                for (size_t j = 0; j < DIM; ++j)
                {
                        vec[j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
                        n2 += vec[j] * vec[j];
                }
                const float inv = 1.0f / std::sqrt(n2);
                for (float& v : vec) v *= inv;
                big.add_vector(vec.data(), DIM, "doc" + std::to_string(i));
        }
        std::vector<float> qr(DIM);
        {
                float n2 = 0.0f;
                for (size_t j = 0; j < DIM; ++j)
                {
                        qr[j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
                        n2 += qr[j] * qr[j];
                }
                const float inv = 1.0f / std::sqrt(n2);
                for (float& v : qr) v *= inv;
        }
        const size_t K = 5;
        auto t0 = std::chrono::steady_clock::now();
        auto top1 = big.search_top_k(qr.data(), K);
        auto t1 = std::chrono::steady_clock::now();
        auto top2 = big.search_top_k_par(qr.data(), K);
        auto t2 = std::chrono::steady_clock::now();
        
        const double ms_s = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double ms_p = std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        bool same = (top1.size() == top2.size());
        for (size_t i = 0; same && i < top1.size(); ++i)
                if (top1[i].index != top2[i].index || std::abs(top1[i].score - top2[i].score) > 1e-6f)
                        same = false;
        
        std::cout << "\n[BENCH] n=" << N << " k=" << K
                  << " serial=" << ms_s << " ms  parallel=" << ms_p << " ms"
                  << "  speedup=" << (ms_s / ms_p) << "x"
                  << "  equal=" << (same ? "yes" : "NO!") << "\n";
        for (size_t i = 0; i < top1.size() && i < 5; ++i)
                std::cout << "  top[" << i << "] idx=" << top1[i].index
                          << " score=" << top1[i].score << "\n";
        
        // LFU-ВЫСЕЛЕНИЕ (демо на игрушечном индексе: 10 векторов → 5 строк лога
        // вместо 6800; основной индекс не трогается, и работает мгновенно)
        {
                RAG demo_rag(hybrid.dim());
                std::vector<float> dv(hybrid.dim());
                std::srand(11);
                for (size_t i = 0; i < 10; ++i)
                {
                        for (float& v : dv)
                                v = static_cast<float>(rand()) / RAND_MAX - 0.5f;
                        demo_rag.add(dv.data(), dv.size(), "toy" + std::to_string(i));
                }
                demo_rag.set_limit(5);
                while (demo_rag.size() > 5)
                        demo_rag.evict();
                std::cout << "[FILTER] После лимита 5 осталось чанков: " << demo_rag.size() << "\n";
        }
        {
                const size_t d = hybrid.dim();
                std::vector<int8_t> q(d);
                float scale = 1.0f, zero = 0.0f;
                quantize_to_int8(hybrid.dense().vector(0), d, q.data(), scale, zero);
                std::vector<float> back(d);
                dequantize_from_int8(q.data(), d, scale, zero, back.data());
                float max_err = 0.0f, sq = 0.0f;
                for (size_t i = 0; i < d; ++i)
                {
                        const float e = std::abs(back[i] - hybrid.dense().vector(0)[i]);
                        max_err = std::max(max_err, e);
                        sq += e * e;
                }
                const float rmse = std::sqrt(sq / d);
                std::cout << "[QUANT] float(" << (4 * d) << " Б) -> int8(" << (1 * d) << " Б) = x4  max_err=" << max_err << "  rmse=" << rmse << "\n";
        }
        // ==== Интерактивная консоль: вопрос или /команда ====
        std::cout << "\n[CONSOLE] Вводи запрос или команду: /stats, /bench N, /quit\n";
        std::string line;
        size_t read_lines = 0;   // сколько строк реально пришло с ввода
        while (std::getline(std::cin, line))
        {
                ++read_lines;
                // Из пайпа/консоли строка может прийти с \r\n и хвостовыми пробелами
                // (cmd echo даже добавляет пробел в конец). Срезаем весь хвостовой мусор,
                // иначе "/quit " не совпадёт с "/quit" и REPL не выйдет из цикла.
                while (!line.empty() &&
                       (line.back() == '\r' || line.back() == '\n' ||
                        line.back() == ' ' || line.back() == '\t'))
                        line.pop_back();
                if (line == "/quit")
                        break;
                if (line == "/stats")
                {
                        std::cout << "[STATS] чанков=" << hybrid.size()
                                  << " dim=" << hybrid.dim() << "\n";
                        continue;
                }
                if (line.rfind("/bench", 0) == 0)
                {
                        size_t bn = 20000;
                        if (line.size() > 6)
                                bn = static_cast<size_t>(std::atoll(line.c_str() + 6));
                        constexpr size_t BDIM = 384;
                        constexpr size_t BK = 5;
                        VectorStorage bbig;
                        bbig.dim = BDIM;
                        std::vector<float> bv(BDIM);
                        std::srand(7);
                        for (size_t i = 0; i < bn; ++i)
                        {
                                float n2 = 0.0f;
                                for (size_t j = 0; j < BDIM; ++j)
                                {
                                        bv[j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
                                        n2 += bv[j] * bv[j];
                                }
                                const float inv = 1.0f / std::sqrt(n2);
                                for (float& v : bv) v *= inv;
                                bbig.add_vector(bv.data(), BDIM, "doc" + std::to_string(i));
                        }
                        std::vector<float> bqr(BDIM);
                        {
                                float n2 = 0.0f;
                                for (size_t j = 0; j < BDIM; ++j)
                                {
                                        bqr[j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
                                        n2 += bqr[j] * bqr[j];
                                }
                                const float inv = 1.0f / std::sqrt(n2);
                                for (float& v : bqr) v *= inv;
                        }
                        auto bt0 = std::chrono::steady_clock::now();
                        auto b1 = bbig.search_top_k(bqr.data(), BK);
                        auto bt1 = std::chrono::steady_clock::now();
                        auto b2 = bbig.search_top_k_par(bqr.data(), BK);
                        auto bt2 = std::chrono::steady_clock::now();
                        const double bs = std::chrono::duration<double, std::milli>(bt1 - bt0).count();
                        const double bp = std::chrono::duration<double, std::milli>(bt2 - bt1).count();
                        std::cout << "[BENCH] n=" << bn << " serial=" << bs
                                  << " ms parallel=" << bp << " ms speedup=" << (bs / bp) << "x\n";
                        continue;
                }
                // иначе — гибридный поиск через фасад (dense + BM25 + RRF внутри)
                // Эхо запроса: в пайпе (echo ...) ввод не отображается,
                // без этой строки результаты непонятно к какому вопросу относятся.
                std::cout << "\n[HYBRID] Запрос: \"" << line << "\"\n";
                auto merged = hybrid.search(line, 3);
                if (merged.empty() || merged[0].rrf < 0.015f) {
                    std::cout << "  [ничего не найдено]\n";
                } else {
                    for (const HybridHit& h : merged)
                        std::cout << "  [" << h.index << "] rrf=" << h.rrf
                                  << " " << hybrid.text(h.index) << "\n";
                }
        }
        // Ввод закончился, но ни одной строки не пришло (мгновенный EOF) —
        // значит это запуск из панели IDE/редиректа, где клавиатуры нет вообще.
        if (read_lines == 0)
        {
                std::cout << "\n[CONSOLE] stdin не является живой консолью — ввод закрыт сразу.\n"
                          << "[CONSOLE] Это нормально для панели Run в IDE (клавиатура туда не идёт).\n"
                          << "[CONSOLE] Чтобы сидеть и вводить:\n"
                          << "[CONSOLE]   1) открой cmd или вкладку Terminal в IDE;\n"
                          << "[CONSOLE]   2) запусти exe оттуда, либо двойной клик по exe в Explorer.\n";
        }
        else
        {
                std::cout << "\n[CONSOLE] Сеанс завершён.\n";
        }
        // Чтобы окно не пропадало мгновенно — держим его, если есть настоящая
        // консоль (иначе это пайп/IDE, и висеть на ожидании клавиши нельзя).
        if (stdout_is_tty())
                console_pause();
        return 0;
}