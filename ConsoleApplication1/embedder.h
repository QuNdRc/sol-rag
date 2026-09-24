#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include "rag_core.h" // интерфейс Embedder

// Лексический эмбеддер: BoW -> TF-IDF с русским стеммингом и стоп-словами.
// Реализует общий интерфейс Embedder, поэтому подставляется в RAG/HybridRag
// наравне с LlamaCppEmbedder.
class TfIdfEmbedder : public Embedder
{
public:
    // первый проход, строит словарь и idf по всему корпусу dim = размер словаря
    void fit(const std::vector<std::string>& corpus);
    //чанк или вопрос, вектор длиной vocab_size (сырые tf*idf; нормировку делает хранилище)
    std::vector<float> embed(std::string_view text) const override;
    size_t vocab_size() const {return token_to_id_.size();}
    bool save(const std::string& path) const;
    bool load(const std::string& path);
    
private:
    std::vector<std::string> id_to_token_; // id -> токен(дамп\дебаг)
    std::unordered_map<std::string, size_t> token_to_id_; // токен -> id
    std::vector<float> idf_; // idf по id 
    
    static std::wstring to_wide(const std::string& utf8);
    static std::string to_utf8(const std::wstring& ws);
    static std::vector<std::wstring> tokenize(const std::wstring& ws);
};