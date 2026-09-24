#include <cmath>
#include <utility>
#include <queue>
#include <algorithm>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include "glav.h"

struct MinScoreCmp
{
    bool operator()(const Match& a, const Match& b) const
    {
        return a.score > b.score;
    }
};

bool VectorStorage::add_vector(const float* vec, size_t vec_dim, const std::string& text)
{
    if (vec_dim != dim) 
        return false; //нельзя смешивать размерности в одном буфере
    //1)хитрость, нормализуем вектор на входе. тогда косинус = скалярное произведение, и поиск обходится без sqrt на горячем пути
    float norm2 = 0.0f;
    for (size_t i = 0; i < vec_dim; ++i)
        norm2 += vec[i] * vec[i];
    
    const float norm = std::sqrt(norm2);
    if (norm2 < 1e-8f)
        return false; //нулевой вектор по нему искать нечего
        
    //2) одна аллокация заранее вместо перераспределений на каждый push 
    all_dimensions.reserve(all_dimensions.size() + vec_dim);
    const float inv = 1.0f / norm;
    for (size_t i = 0; i < vec_dim; ++i)
        all_dimensions.push_back(vec[i] * inv);
    
    //3) метаданные параллельны координатам raw_texts[i] <-> координаты [i*dim, (i+1)*dim]
    raw_texts.push_back(text);
    access_counters.push_back(0u);
    
    if (limit_ != 0)
        while (vector_count() < limit_)
            evict_lfu(); // переполнение -> выбиваем холодного (LFU) 
    
    return true;
    
}

float VectorStorage::dot_product(const float* a, const float* b, size_t d) const
{
    float sum = 0.0f;
    for (size_t i = 0; i < d; ++i)
        sum += a[i] * b[i];
    return sum;
}

std::vector<Match> VectorStorage::search_top_k(const float* query, size_t k)
{
    const size_t n = vector_count();
    if (n == 0 || k == 0)
        return {};
    if (k > n)
        k = n;
    
    //запрос пришел сырым - нормализуем один раз, база у нас уже нормализована
    
    float q_norm2 = 0.0f;
    for (size_t i = 0; i < dim; ++i)
        q_norm2 += query[i] * query[i];
    const float q_norm = std::sqrt(q_norm2);
    if (q_norm2 < 1e-8f)
        return {};
    
    std::vector<float> q(query, query + dim);
    for (float& v : q)
        v /= q_norm;
    
    //min-heap: O(n*log k), память O(k), инкрементально (готово под Day 2: куча на поток + merge)
        
    std::priority_queue<Match, std::vector<Match>, MinScoreCmp> heap;
    for (size_t i = 0; i < n; ++i)
    {
        const float score = dot_product(q.data(), &all_dimensions[i * dim], dim);
        if (score <= 0.0f) // нет пересечения смыслов - в топ не идем
            continue;
        
        if (heap.size() < k || score > heap.top().score)
        {
            ++access_counters[i]; // важность (реально был в актуальном топе)
            if (heap.size() < k)
                heap.push(Match{i, score});
            else
            {
                heap.pop(); // // O(log k) — выкинули худшего
                heap.push(Match{i, score});
            }
        }
    }
    std::vector<Match> top;
    top.reserve(heap.size());
    while (!heap.empty())
    {
        top.push_back(heap.top());
        heap.pop();
    }
    std::reverse(top.begin(), top.end()); // top[0] - лучший
    return top;
}

std::vector<Match> VectorStorage::search_top_k_par(const float* query, size_t k)
{
    const size_t n = vector_count();
    if (n == 0 || k == 0)
        return {};
    if (k > n)
        k = n;
    
    // запрос нормализуем один раз до потоков, общий read-only буфер
    float q_norm2 = 0.0f;
    for (size_t i = 0; i < dim; ++i)
        q_norm2 += query[i] * query[i];
    const float q_norm = std::sqrt(q_norm2);
    if (q_norm2 < 1e-8f)
        return {};
    std::vector<float> q(query, query + dim);
    for (float& v : q)
        v /= q_norm;
    
    // делим базу на P диапозонов (континуум -> плюс к кэш-лояльности)
    
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    const size_t P = hw;
    const size_t slice = (n + P - 1) / P;
    
    std::vector<std::vector<Match>> partial(P); // топ k каждой полосы 
    std::vector<long long> part_us(P, 0); // микросекунды на полосу
    std::vector<std::jthread> workers;
    for (size_t t = 0; t < P; ++t)
    {
        const size_t begin = t * slice;
        const size_t end = std::min(begin + slice, n);
        if (begin >= end)
            continue; // полоса пустая (P бывает > n) - поток не заводим
        workers.emplace_back([this, &q, k, t, begin, end, &partial, &part_us]()
        {
            const auto t_c0 = std::chrono::high_resolution_clock::now();
            std::priority_queue<Match, std::vector<Match>, MinScoreCmp> heap;
            for (size_t i = begin; i < end; ++i)
            {
                const float score = dot_product(q.data(), &all_dimensions[i * dim], dim);
                if (score <= 0.0f)
                    continue;
                if (heap.size() < k || score > heap.top().score)
                {
                    ++access_counters[i]; // только свой диапозон -> гонок нет
                    if (heap.size() < k)
                        heap.push(Match{i, score});
                    else
                    {
                        heap.pop();
                        heap.push(Match{i, score});
                    }
                }
            }
            while (!heap.empty()) // выгружаем свою кучу в partial[t]
            {
                partial[t].push_back(heap.top());
                heap.pop();
            }
            const auto t_c1 = std::chrono::steady_clock::now();
            part_us[t] = std::chrono::duration_cast<std::chrono::microseconds>(t_c1 - t_c0).count();
        });
    }
    for (auto& w : workers)
        w.join(); // jthread саь бы join-нул при разрушении, явно тоже ок 
    for (size_t t = 0; t < P; ++t)
    {
        const size_t begin = t * slice;
        const size_t end = std::min(begin + slice, n);
        if (begin >= end)
            std::cout << "[LOG] Поток #" << t << " обработал векторы " << begin << "-" << end << " за " << part_us[t] << " мкс\n";
    }
    // слияние P частичных топ-k в один глобальный 
    std::priority_queue<Match, std::vector<Match>, MinScoreCmp> heap;
    for (size_t t = 0; t < P; ++t)
        for (size_t j = 0; j < partial[t].size(); ++j)
        {
            const Match& m = partial[t][j];
            if (heap.size() < k || m.score > heap.top().score)
            {
                if (heap.size() < k)
                    heap.push(m);
                else
                {
                    heap.pop();
                    heap.push(m);
                }
            }
        }
    std::vector<Match> top;
    top.reserve(heap.size());
    while (!heap.empty())
    {
        top.push_back(heap.top());
        heap.pop();
    }
    std::reverse(top.begin(), top.end()); // top[0] - лучший 
    return top;
}

void VectorStorage::set_limit (size_t max_vectors)
{
    limit_ = max_vectors;
}
void VectorStorage::evict_lfu()
{
    const size_t n = vector_count();
    if (n == 0) return;
    // ищем чанк с минимальным access_counters 
    size_t worst = 0;
    for (size_t i = 0; i < n; ++i)
        if (access_counters[i] < access_counters[worst])
            worst = i;
    
    all_dimensions.erase(all_dimensions.begin() + worst * dim, all_dimensions.begin() + (worst + 1) * dim);
    raw_texts.erase(raw_texts.begin() + worst);
    std::cout << "[FILTER] Выселен вектор #" << worst << " (low attention weight)\n";
}

void quantize_to_int8(const float* v, size_t d, int8_t* out, float& scale, float& zero)
{
    float mn = v[0], mx = v[0];
    for (size_t i = 1; i < d; ++i)
    {
        mn = std::min(mn, v[i]);
        mx = std::max(mx, v[i]);
    }
    const float range = mx - mn;
    scale = (range < 1e-20f) ? 1.0f : range / 254.0f; // 254 шага между -127...127
    zero = (mn + mx) * 0.5f;
    for (size_t i = 0; i < d; ++i)
    {
        const int qi = static_cast<int>(std::lround((v[i] - zero) / scale));
        out[i] = static_cast<int8_t>(std::clamp(qi, -127, 127));
    }
}

void dequantize_from_int8(const int8_t* q, size_t d, float scale, float zero, float* out)
{
    for (size_t i = 0; i < d; ++i)
        out[i] = zero + scale * static_cast<float>(q[i]);
}

bool VectorStorage::save(const std::string& path) const
{
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    
    uint64_t d = dim, n = vector_count();
    out.write(reinterpret_cast<const char*>(&d), sizeof(d));
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(all_dimensions.data()), all_dimensions.size() * sizeof(float));
    
    for (const auto& t : raw_texts)
    {
        uint64_t len = t.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(t.data(), static_cast<std::streamsize>(t.size()));
    }
    out.write(reinterpret_cast<const char*>(access_counters.data()), access_counters.size() * sizeof(uint32_t));
    return true;
}

bool VectorStorage::load(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint64_t d = 0, n = 0;
    in.read(reinterpret_cast<char*>(&d), sizeof(d));
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!in || d == 0) return false;
    
    all_dimensions.resize(static_cast<size_t>(n) * d);
    in.read(reinterpret_cast<char*>(all_dimensions.data()), all_dimensions.size() * sizeof(float));
    
    raw_texts.resize(n);
    for (auto& t : raw_texts)
    {
        uint64_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        t.resize(len);
        in.read(&t[0], static_cast<std::streamsize>(len));
    }
    access_counters.resize(n);
    in.read(reinterpret_cast<char*>(access_counters.data()), n * sizeof(uint32_t));
    
    dim = d;
    return in.good();
}