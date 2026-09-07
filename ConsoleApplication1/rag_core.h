#pragma once 

#include <vector>
#include <string>
#include <string_view>
#include <cstddef>
#include "kakayatohuinya.h"

// ---- Адаптеры (прикладной уровень, ВНЕ ядра) ----
// Куда потребитель втыкает свою реализацию (BoW сейчас, llama.cpp позже, свои Chunker'ы).

struct Embedder
{
    virtual ~Embedder() = default;
    virtual std::vector<float> embed(std::string_view text)
    const = 0;
};

struct Chunker
{
    virtual ~Chunker() = default;
    virtual std::vector<std::string> chunk(const std::string& doc) const = 0;
};
// ---- Публичный контракт ядра ----
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
