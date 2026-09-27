#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <cstddef>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include "glav.h"
#include "bm25_index.h"

//  Адаптеры (прикладной уровень, ВНЕ ядра)
// Куда потребитель втыкает свою реализацию (BoW сейчас, llama.cpp позже, свои Chunker'ы).

struct Embedder
{
    virtual ~Embedder() = default;
    virtual std::vector<float> embed(std::string_view text) const = 0;
};

struct Chunker
{
    virtual ~Chunker() = default;
    virtual std::vector<std::string> chunk(const std::string& doc) const = 0;
};

// Ядро не знает про язык/эмбеддер: на вход готовый вектор.
// Буферы спрятаны — инвариант размера не сломать снаружи.
class RAG
{
public:
    explicit RAG(size_t dim)
    {
        store_.dim = dim;
    }
    bool add(const float* vec, size_t vec_dim, std::string_view text)
    {
        return store_.add_vector(vec, vec_dim, std::string(text));
    }
    void set_limit(size_t n) { store_.set_limit(n); }
    void evict() {store_.evict_lfu(); }
    bool save(const std::string& path) const { return store_.save(path);}
    bool load(const std::string& path) { return store_.load(path); }
    std::vector<Match> search(const float* query, size_t k)
    {
        return store_.search_top_k(query, k);
    }
    std::vector<Match> search_par(const float* query, size_t k)
    {
        return store_.search_top_k_par(query, k);
    }
    std::string_view text(size_t i) const { return store_.raw_texts[i]; }
    size_t size() const noexcept { return store_.vector_count(); }
    size_t dim() const noexcept { return store_.dim; }
    const float* vector(size_t i) const { return &store_.all_dimensions[i * store_.dim]; }
private:
    VectorStorage store_; // движок внутри
};

// Гибридный поиск: dense (RAG) + sparse (BM25) 
// Результат слияния: индекс чанка + оценка RRF.
struct HybridHit
{
    size_t index;
    float  rrf;
};

// Reciprocal Rank Fusion: объединяет два ранжированных списка в один.
// Формула: score(d) = sum по спискам 1 / (k_rrf + rank(d)), k_rrf = 60 (стандарт).
// dense: результаты поиска по эмбеддингам; sparse: результаты BM25.
inline std::vector<HybridHit> rrf_merge(const std::vector<Match>& dense,
                                        const std::vector<std::pair<size_t, float>>& sparse,
                                        size_t k = 3, float k_rrf = 60.0f)
{
    std::unordered_map<size_t, float> scores;
    for (size_t i = 0; i < dense.size(); ++i)
        scores[dense[i].index] += 1.0f / (k_rrf + static_cast<float>(i + 1));
    for (size_t i = 0; i < sparse.size(); ++i)
        scores[sparse[i].first] += 1.0f / (k_rrf + static_cast<float>(i + 1));

    std::vector<HybridHit> merged;
    merged.reserve(scores.size());
    for (const auto& [idx, score] : scores)
        merged.push_back({idx, score});

    const size_t n = std::min(k, merged.size());
    std::partial_sort(merged.begin(), merged.begin() + static_cast<std::ptrdiff_t>(n),
                      merged.end(),
                      [](const HybridHit& a, const HybridHit& b) { return a.rrf > b.rrf; });
    merged.resize(n);
    return merged;
}

//  Фасад гибридного поиска
// Держит dense-индекс (RAG + Embedder) и sparse-индекс (BM25) В СИНХРОНЕ:
// add_chunk() кладёт чанк в оба, search() ищет по обоим и сливает через RRF.
// Потребителю не нужно знать про два индекса — один класс, два метода.
//
// Использование:
//   HybridRag hr(emb, emb.dim());            // emb — любой Embedder (BGE-M3, TF-IDF...)
//   hr.add_chunk("текст чанка");             // при индексации
//   auto hits = hr.search("запрос", 3);      // при поиске
//   hr.text(hits[0].index);                  // текст лучшего чанка
class HybridRag
{
public:
    // dim — размерность вектора эмбеддера (должна совпадать у всех чанков).
    HybridRag(Embedder& emb, size_t dim) : emb_(&emb), rag_(dim) {}

    // Положить чанк в оба индекса. Если эмбеддер вернул пустой вектор
    // (текст длиннее контекста модели и т.п.) — чанк всё равно кладётся
    // в BM25, а в dense попадает нуль-вектор через add(), иначе индексы
    // разъедутся по id и text(i) будет читать не тот чанк.
    // Возвращает false, только если эмбеддинг не удалось получить И
    // dense-добавление отвергло вектор.
    bool add_chunk(std::string_view text)
    {
        std::vector<float> emb = emb_->embed(text);
        bool ok = true;
        if (emb.size() == rag_.dim())
        {
            ok = rag_.add(emb.data(), emb.size(), text);
        }
        else
        {
            // Пустой/битый эмбеддинг: нуль-вектор. add_vector отвергает ноль
            // (нормировка), поэтому кладём eps-вектор — он не матчит ни с чем
            // осмысленно, но держит нумерацию индексов в синхроне с BM25.
            std::vector<float> dummy(rag_.dim(), 0.0f);
            dummy[0] = 1e-6f;
            ok = rag_.add(dummy.data(), dummy.size(), text);
        }
        bm25_.add_document(text);
        return ok;
    }

    // Гибридный поиск: top-k_dense + top-k_sparse -> RRF -> top-k.
    // Не const: RAG::search обновляет access_counters (LFU-статистика).
    std::vector<HybridHit> search(std::string_view query, size_t k = 3)
    {
        std::vector<float> qv = emb_->embed(query);
        if (qv.size() != rag_.dim())
            qv.assign(rag_.dim(), 0.0f);
        auto dense  = rag_.search(qv.data(), k + 2);   // берём с запасом для слияния
        auto sparse = bm25_.search(query, k + 2);
        return rrf_merge(dense, sparse, k);
    }

    // Доступ к тексту чанка по индексу (единая нумерация dense и BM25).
    std::string_view text(size_t i) const { return rag_.text(i); }

    size_t size() const noexcept { return rag_.size(); }
    size_t dim() const noexcept { return rag_.dim(); }

    // Персистентность dense-индекса. BM25-индекс не сохраняется —
    // он пересобирается из текстов при load (быстро, O(n)).
    bool save(const std::string& path) const { return rag_.save(path); }
    bool load(const std::string& path)
    {
        if (!rag_.load(path)) return false;
        bm25_ = Bm25Index{};
        for (size_t i = 0; i < rag_.size(); ++i)
            bm25_.add_document(rag_.text(i));
        return true;
    }

    // Доступ к внутренностям (для бенчмарков/демо LFU и квантизации).
    RAG&       dense()       { return rag_; }
    const RAG& dense() const { return rag_; }

private:
    Embedder*  emb_;
    RAG        rag_;
    Bm25Index  bm25_;
};
