#pragma once
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <cstdint>

/// Компактный BM25-индекс: инвертированный индекс + ранжирование.
/// Токенизация: грубая (все не-ASCII байты считаются буквами — хватает
/// для русского/английского в UTF-8). Без стемминга (скорость > точность).
struct Bm25Index {
    float k1 = 1.2f;
    float b  = 0.75f;

    std::vector<size_t> doc_lengths;
    float avgdl = 0.0f;
    size_t doc_count = 0;

    // posting_lists[term] = [{doc_id, tf}, ...]
    std::unordered_map<std::string, std::vector<std::pair<size_t, uint16_t>>> posting_lists;
    std::unordered_map<std::string, size_t> df; // document frequency

    static bool is_letter_byte(unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= 0x80);
    }

    static std::vector<std::string> tokenize(std::string_view text) {
        std::vector<std::string> tokens;
        std::string cur;
        for (size_t i = 0; i < text.size(); ) {
            unsigned char c = (unsigned char)text[i];
            if (is_letter_byte(c)) {
                // tolower только для ASCII (a-z, A-Z), не для UTF-8 (>= 0x80)
                cur += (c >= 0x80) ? (char)c : (char)std::tolower(c);
                ++i;
                while (i < text.size() && ((unsigned char)text[i] & 0xC0) == 0x80) {
                    c = (unsigned char)text[i];
                    cur += (c >= 0x80) ? (char)c : (char)std::tolower(c);
                    ++i;
                }
            } else {
                if (cur.size() >= 2) tokens.push_back(cur);
                cur.clear();
                ++i;
            }
        }
        if (cur.size() >= 2) tokens.push_back(cur);
        return tokens;
    }

    void add_document(std::string_view text) {
        auto tokens = tokenize(text);
        doc_lengths.push_back(tokens.size());
        ++doc_count;

        std::unordered_map<std::string, uint16_t> tf;
        for (const auto& t : tokens) tf[t]++;

        size_t doc_id = doc_count - 1;
        for (const auto& [term, freq] : tf) {
            posting_lists[term].push_back({doc_id, freq});
            df[term]++;
        }

        avgdl = 0.0f;
        for (auto len : doc_lengths) avgdl += (float)len;
        if (doc_count > 0) avgdl /= (float)doc_count;
    }

    float score_doc(size_t doc_id, const std::vector<std::string>& qterms) const {
        float score = 0.0f;
        size_t dl = doc_lengths[doc_id];
        for (const auto& term : qterms) {
            auto it_df = df.find(term);
            if (it_df == df.end()) continue;
            float idf = std::log(1.0f + ((float)doc_count - (float)it_df->second + 0.5f)
                                 / ((float)it_df->second + 0.5f));
            auto it_pl = posting_lists.find(term);
            if (it_pl == posting_lists.end()) continue;
            uint16_t tf = 0;
            for (const auto& [did, f] : it_pl->second)
                if (did == doc_id) { tf = f; break; }
            if (tf == 0) continue;
            float num = (float)tf * (k1 + 1.0f);
            float den = (float)tf + k1 * (1.0f - b + b * (float)dl / avgdl);
            score += idf * num / den;
        }
        return score;
    }

    std::vector<std::pair<size_t, float>> search(std::string_view query, size_t k) const {
        auto qterms = tokenize(query);
        if (qterms.empty()) return {};
        std::vector<std::pair<size_t, float>> scored;
        for (size_t i = 0; i < doc_count; ++i) {
            float s = score_doc(i, qterms);
            if (s > 0.0f) scored.push_back({i, s});
        }
        size_t n = std::min(k, scored.size());
        std::partial_sort(scored.begin(), scored.begin() + (std::ptrdiff_t)n,
                          scored.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
        scored.resize(n);
        return scored;
    }
};