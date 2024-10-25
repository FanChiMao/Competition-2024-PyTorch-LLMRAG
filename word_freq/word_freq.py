from collections import Counter
import jieba
import json

def read_faq_data(faq_items: list) -> str:
    text = ""
    for item in faq_items:
        if 'question' in item:
            text += item['question'] + " "
        if 'answers' in item:
            for temp_answer in item['answers']:
                text += temp_answer + " "
    text = ''.join(text.splitlines())
    return text

def remove_stop_word(corpus: list[str]) -> list[str]:
    new_corpus = []
    stop_word = {}.fromkeys([line.strip() for line in open(rb'../reference/stop_word.txt', encoding='utf-8')])
    for word in corpus:
        if word != ' ' and word not in stop_word:
            new_corpus.append(word)

    return new_corpus

if __name__ == '__main__':
    jieba.load_userdict("../reference/dict.txt")

    with open("../corpus_dict/insurance_corpus_dict.json", 'rb') as f:
        corpus_dict_insurance = {int(k): v for k, v in json.load(f).items()}

    # with open("../corpus_dict/finance_corpus_dict.json", 'rb') as f:
    #     corpus_dict_finance = {int(k): v for k, v in json.load(f).items()}

    with open("../corpus_dict/faq_corpus_dict.json", 'rb') as f:
        key_to_source_dict = {int(k): v for k, v in json.load(f).items()}

    # 整合各檔案斷詞結果
    insurance_tokenized_corpus = remove_stop_word(list(jieba.cut(' '.join(corpus_dict_insurance.values()))))
    # finance_tokenized_corpus = remove_stop_word(list(jieba.cut(' '.join(corpus_dict_finance.values()))))
    faq_tokenized_corpus = remove_stop_word(list(jieba.cut(' '.join(key_to_source_dict.values()))))

    # 輸出詞頻率檔案
    with open("word_freq_insurance_dict.json", 'w', encoding='utf8') as f:
        json.dump(Counter(insurance_tokenized_corpus).most_common(), f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符

    # with open("word_freq_finance_dict.json", 'w', encoding='utf8') as f:
    #     json.dump(Counter(finance_tokenized_corpus).most_common(), f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符

    with open("word_freq_faq_dict.json", 'w', encoding='utf8') as f:
        json.dump(Counter(faq_tokenized_corpus).most_common(), f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符

    tokenized_corpus = insurance_tokenized_corpus
    # tokenized_corpus.extend(finance_tokenized_corpus)
    tokenized_corpus.extend(faq_tokenized_corpus)
    dictionary = Counter(tokenized_corpus).most_common()

    # 輸出詞頻率檔案
    # with open("word_freq_dict.json", 'w', encoding='utf8') as f:
    #     json.dump(dictionary, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符