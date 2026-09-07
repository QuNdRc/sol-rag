#include <cmath>
#include <unordered_set>
#include <fstream>
#include "embedder.h"

namespace
{
    // Стоп-слова русского (эмбеддер — языковой адаптер, список живёт здесь)
    const std::unordered_set<std::wstring> k_stopwords = {
        L"и", L"в", L"во", L"не", L"что", L"он", L"на", L"я", L"с", L"со", L"как", L"а",
        L"то", L"все", L"она", L"так", L"его", L"но", L"да", L"ты", L"к", L"у", L"же",
        L"вы", L"за", L"бы", L"по", L"только", L"ее", L"мне", L"было", L"вот", L"от",
        L"меня", L"еще", L"о", L"из", L"ему", L"когда", L"даже", L"ли", L"если", L"уже",
        L"или", L"ни", L"быть", L"была", L"был", L"до", L"вас", L"уж", L"вам", L"там",
        L"потом", L"себя", L"ей", L"может", L"они", L"тут", L"где", L"есть", L"надо",
        L"для", L"мы", L"тебя", L"их", L"чем", L"сам", L"без", L"чего", L"раз", L"тоже",
        L"себе", L"под", L"тогда", L"кто", L"этот", L"того", L"потому", L"этого", L"какой",
        L"здесь", L"этом", L"один", L"почти", L"тем", L"чтобы", L"нее", L"сейчас", L"были",
        L"куда", L"зачем", L"всех", L"никогда", L"можно", L"при", L"об", L"другой", L"хоть",
        L"после", L"больше", L"тот", L"через", L"эти", L"нас", L"про", L"всего", L"них",
        L"много", L"эту", L"перед", L"иногда", L"лучше", L"чуть", L"нельзя", L"такой",
        L"им", L"более", L"всегда", L"конечно", L"всю", L"должен", L"должна", L"должно",
        L"должны"
    };
    // Лёгкий стемминг: режем длиннейшее из известных окончаний (1-2 шт.)
    struct { const wchar_t* s; size_t n; } k_suf[] = {
        {L"ами", 3}, {L"ями", 3}, {L"ого", 3}, {L"его", 3}, {L"ую", 3}, {L"юю", 3},
        {L"ых", 2}, {L"их", 2}, {L"ым", 2}, {L"им", 2}, {L"ом", 2}, {L"ем", 2},
        {L"ый", 2}, {L"ий", 2}, {L"ой", 2}, {L"ая", 2}, {L"яя", 2},
        {L"ов", 2}, {L"ев", 2}, {L"ах", 2}, {L"ям", 2}, {L"ам", 2},
        {L"а", 1}, {L"я", 1}, {L"ы", 1}, {L"и", 1}, {L"у", 1}, {L"ю", 1},
        {L"е", 1}, {L"о", 1}, {L"ь", 1}
    };

    std::wstring stem_ru(std::wstring w)
    {
        if (w.size() <= 3)
            return w; // короткие (сп, не) не трогаем
        for (auto& p : k_suf)
        {
            if (w.size() > p.n && w.compare(w.size() - p.n, p.n, p.s) == 0)
            {
                w.resize(w.size() - p.n);
                break; // срезали первое (самое длинное) подошедшее окончание
            }
        }
        return w;
    }

}

std::wstring TfIdfEmbedder::to_wide(const std::string& utf8)
{
    std::wstring out; 
    out.reserve(utf8.size());
    for (size_t i = 0; i < utf8.size();)
    {
        unsigned char c = static_cast<unsigned char>(utf8[i]);
        if ((c & 0x80) == 0)
        {
            out += static_cast<wchar_t>(c);
            ++i;
        }
        else if ((c & 0xE0) == 0xC0 && i + 1 < utf8.size())
        {
            out += static_cast<wchar_t>(((c & 0x1F) << 6) | (static_cast<unsigned char>(utf8[i + 1]) & 0x3F));
            i += 2;
        }
        else if ((c & 0xF0) == 0xE0 && i + 2 < utf8.size())
        {
            out += static_cast<wchar_t>(((c & 0x0F) << 12) | ((static_cast<unsigned char>(utf8[i + 1]) & 0x3F) << 6) | (static_cast<unsigned char>(utf8[i + 2]) & 0x3F));
            i += 3;
        }
        else if ((c & 0xF8) == 0xF0 && i + 3 < utf8.size())
        {
            unsigned long cp = ((c & 0x07) << 18) | ((static_cast<unsigned char>(utf8[i + 1]) & 0x3F) << 12) | ((static_cast<unsigned char>(utf8[i + 2]) & 0x3F) << 6) | (static_cast<unsigned char>(utf8[i + 3]) & 0x3F);
            if (cp >= 0x10000)
            {
                cp -= 0x10000;
                out += static_cast<wchar_t>(0xD800 + (cp >> 10));
                out += static_cast<wchar_t>(0xDC00 + (cp & 0x3FF));
            }
            i += 4;
        }
        else 
            ++i;
    }
    return out;
}

std::string TfIdfEmbedder::to_utf8(const std::wstring& ws)
{
    std::string out;
    for (size_t i = 0; i < ws.size(); ++i) // старший суррогат
    {
        unsigned long cp = ws[i];
        if (cp >= 0xD800 && cp <= 0xD8FF && i + 1 < ws.size())
        {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (ws[i + 1] - 0xDC00);
            ++i;
        }
        if (cp < 0x80) out += static_cast<char>(cp);
        else if (cp < 0x800)
        {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
        else if (cp < 0x10000)
        {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
        else
        {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    return out;
}
std::vector<std::wstring> TfIdfEmbedder::tokenize(const std::wstring& ws)
{
    std::vector<std::wstring> tokens;
    std::wstring cur;
    
    auto is_letter = [](wchar_t c)
    {
        return (c >= L'a' && c <= L'z') || (c >= L'A' && c <= L'Z') ||
                   (c >= L'0' && c <= L'9') ||
                   (c >= L'\u0430' && c <= L'\u044F') ||  // а..я
                   (c >= L'\u0410' && c <= L'\u042F');
    };
    
    for (wchar_t ch : ws)
    {
        wchar_t c = ch; 
        if (c == L'\u0401' || c == L'\u0451') c = L'\u0435'; // ё/Ё -> е
        else if (c >= L'\u0410' && c <= L'\u042F') c += 0x20; //кириллица верх -> низ
        if (c >= L'A' && c <= L'Z') c += 0x20; // ASCII верх -> низ
        if (is_letter(c))
            cur += c;
        else if (!cur.empty())
        {
            if (cur.size() >= 2 && k_stopwords.find(cur) == k_stopwords.end())
            {
                std::wstring st = stem_ru(cur);
                if (st.size() >= 2)
                    tokens.push_back(st);
            }
            cur.clear();
        }
    }
    if (cur.size() >= 2 && k_stopwords.find(cur) == k_stopwords.end())
    {
        std::wstring st = stem_ru(cur);
        if (st.size() >= 2)
            tokens.push_back(st);
    }
       
    return tokens;
}

void TfIdfEmbedder::fit(const std::vector<std::string>& corpus)
{
    const size_t D = corpus.size();
    std::unordered_map<std::string, size_t> df; // токен -> сколько чанков его содержат
    for (size_t i = 0; i < D; ++i)
    {
        std::vector<std::wstring> toks = tokenize(to_wide(corpus[i]));
        std::unordered_set<std::string> uniq; // уникальные токены чанка, нужно для df 
        for (size_t j = 0; j < toks.size(); ++j)
            uniq.insert(to_utf8(toks[j]));
        
        for (auto it = uniq.begin(); it != uniq.end(); ++it)
        {
            ++df[*it];
            if (token_to_id_.find(*it) == token_to_id_.end())
            {
                token_to_id_[*it] = token_to_id_.size(); // id по первому вхождению
                id_to_token_.push_back(*it);
            }
        }
    }
    //idf(t) = ln ( (1+D) / (1+df(t)) ) + 1
    idf_.resize(token_to_id_.size());
    for (auto it = token_to_id_.begin(); it != token_to_id_.end(); ++it)
        idf_[it->second] = std::logf((1.0f + (float)D) / (1.0f + (float)df[it->first])) + 1.0f;
}

std::vector<float> TfIdfEmbedder::embed(const std::string& text) const
{
    std::vector<float> vec(token_to_id_.size(), 0.0f);
    std::vector<std::wstring> toks = tokenize(to_wide(text));
    std::unordered_map<size_t, size_t> tf; // id -> частота в тексте
    for (size_t j = 0; j < toks.size(); ++j)
    {
        auto it = token_to_id_.find(to_utf8(toks[j])); // OOV токены, пропускаем
        if (it != token_to_id_.end())
            ++tf[it->second];
    }
    for (auto it = tf.begin(); it != tf.end(); ++it)
        vec[it -> first] = static_cast<float>(it -> second) * idf_[it -> first];
    return vec;
}

bool TfIdfEmbedder::save(const std::string& path) const
{
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    uint64_t v = id_to_token_.size();
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    for (const auto& t : id_to_token_)
    {
        uint64_t len = t.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(t.data(), static_cast<std::streamsize>(t.size()));
    }
    out.write(reinterpret_cast<const char*>(idf_.data()), idf_.size() * sizeof(float));
    return true;
}

bool TfIdfEmbedder::load(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint64_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    id_to_token_.resize(v);
    token_to_id_.clear();
    for (uint64_t i = 0; i < v; ++i)
    {
        uint64_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        id_to_token_[i].resize(len);
        in.read(&id_to_token_[i][0], static_cast<std::streamsize>(len));
        token_to_id_[id_to_token_[i]] = static_cast<size_t>(i); // обратный словарь заново
    }
    idf_.resize(v);
    in.read(reinterpret_cast<char*>(idf_.data()), v * sizeof(float));
    return in.good();
}