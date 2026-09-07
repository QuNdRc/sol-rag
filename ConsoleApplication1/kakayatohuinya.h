#pragma once 

#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>

struct Match
{
    size_t index = 0;
    float score = 0.0f;
};



struct VectorStorage
{   
    std::vector<float> all_dimensions;// n * dim float-ов, монолит
    //тексты чанков параллельны индекс вектора *dim соответствует индексу текста
    std::vector<std::string> raw_texts; // метаданные отдельно
    std::vector<uint32_t> access_counters; // "вес" чанка: копим для Day 3 (LFU/H2O)
    
    size_t dim = 384;
    
    //сколько чанков реально лежит
    
    size_t vector_count() const { return all_dimensions.size() / dim; }
    bool add_vector(const float* vec, size_t vec_dim, const std::string& text);
    float dot_product(const float* a, const float* b, size_t d) const;
    std::vector<Match> search_top_k (const float* query, size_t k);
    std::vector<Match> search_top_k_par(const float* query, size_t k);
    size_t limit_ = 0; // 0 = без лимита 
    void set_limit(size_t max_vectors);
    void evict_lfu();
    bool save(const std::string& path) const; 
    bool load(const std::string& path);
};
void quantize_to_int8(const float* v, size_t d, int8_t* out, float& scale, float& zero);
void dequantize_from_int8(const int8_t* q, size_t d, float scale, float zero, float* out);

