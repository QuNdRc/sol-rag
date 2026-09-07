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
#include "kakayatohuinya.h"
#include "rag_core.h"
#include "embedder.h"
#include "llama_embedder.h" 

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
                                if (!p.empty()) chunks.push_back(p);
                }
                std::cout << "[SYSTEM] Чанков из documents/: " << chunks.size() << "\n";
        }

        // ==== СЕМАНТИЧЕСКИЙ RAG (LlamaCppEmbedder, dim=384) ====
        std::filesystem::path exe_dir = std::filesystem::absolute(
            argc > 0 ? argv[0] : ".").parent_path();
        std::filesystem::path model_path = docs.parent_path() / "models"
            / "paraphrase-multilingual-MiniLM-L12-118M-v2-Q8_0.gguf";

        LlamaCppEmbedder sem_emb;
        if (!sem_emb.init(exe_dir.string().c_str(), model_path.string().c_str()))
        {
            std::cout << "[FATAL] LlamaCppEmbedder не загрузился: "
                      << LlamaCppEmbedder::last_error() << "\n";
            return 1;
        }
        std::cout << "[SYSTEM] Семантический эмбеддер загружен, dim="
                  << sem_emb.dim() << "\n";

        const std::string sem_rag_path = (docs / "rag_sem.bin").string();
        RAG sem_rag(sem_emb.dim()); // dim=384

        if (std::filesystem::exists(sem_rag_path))
        {
            if (!sem_rag.load(sem_rag_path))
            {
                std::cout << "[SYSTEM] rag_sem.bin битый\n";
                return 1;
            }
            std::cout << "[SYSTEM] Семантический индекс загружен: "
                      << sem_rag.size() << " чанков, dim=" << sem_rag.dim() << "\n";
        }
        else
        {
            for (size_t i = 0; i < chunks.size(); ++i)
            {
                std::vector<float> emb = sem_emb.embed(chunks[i]);
                sem_rag.add(emb.data(), emb.size(), chunks[i]);
            }
            sem_rag.save(sem_rag_path);
            std::cout << "[SYSTEM] Семантический индекс построен: "
                      << sem_rag.size() << " чанков, dim=" << sem_rag.dim() << "\n";
        }
        
        const std::vector<std::string> queries = {
                "Какая минимальная площадь кухни по нормам?",
                "Какой ширины должна быть лестница?",
        };
        for (const auto& q : queries)
        {
                std::vector<float> qv = sem_emb.embed(q);
                auto top = sem_rag.search(qv.data(), 3);
                std::cout << "\n[SEM] Запрос: \"" << q << "\"\n";
                for (const Match& m : top)
                {
                        std::cout << "  [" << m.index << "] score=" << m.score << " " << sem_rag.text(m.index) << "\n";
                }
        }
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
        
        // LFU-ВЫСЕЛЕНИЕ + КВАНТОВАНИЕ INT8 
        {
                sem_rag.set_limit(5);
                while (sem_rag.size() > 5)
                        sem_rag.evict();
                std::cout << "[FILTER] После лимита 5 осталось чанков: " << sem_rag.size() << "\n";
        }
        {
                const size_t d = sem_rag.dim();
                std::vector<int8_t> q(d);
                float scale = 1.0f, zero = 0.0f;
                quantize_to_int8(sem_rag.vector(0), d, q.data(), scale, zero);
                std::vector<float> back(d);
                dequantize_from_int8(q.data(), d, scale, zero, back.data());
                float max_err = 0.0f, sq = 0.0f;
                for (size_t i = 0; i < d; ++i)
                {
                        const float e = std::abs(back[i] - sem_rag.vector(0)[i]);
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
                        std::cout << "[STATS] чанков=" << sem_rag.size()
                                  << " dim=" << sem_rag.dim() << "\n";
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
                // иначе — запрос к базе знаний
                std::vector<float> qv = sem_emb.embed(line);
                auto top = sem_rag.search(qv.data(), 3);
                for (const Match& m : top)
                        std::cout << "  [" << m.index << "] score=" << m.score
                                  << " " << sem_rag.text(m.index) << "\n";
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