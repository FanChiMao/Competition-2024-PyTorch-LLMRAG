import os
import json
import argparse
import re
from statistics import fmean

from time import perf_counter
from tqdm import tqdm
import jieba  # 用於中文文本分詞
import pdfplumber  # 用於從PDF文件中提取文字的工具
from rank_bm25 import BM25Plus # 使用BM25演算法進行文件檢索

import pytesseract

from text2vec import SentenceModel, semantic_search

embedder = SentenceModel("shibing624/text2vec-base-chinese-sentence")

# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path):
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}
    return corpus_dict

# def load_finance_ocr_data(source_path, corpus_dict: dict = None):
#     if corpus_dict is None: corpus_dict = {}
#     masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
#     for file in tqdm(masked_file_ls):
#         file_id = int(file.replace('.pdf', ''))
#         if file_id not in corpus_dict:
#             corpus_dict[int(file.replace('.pdf', ''))] = read_pdf(os.path.join(source_path, file))
#             with open("corpus_dict/finance_corpus_dict.json", 'w+', encoding='utf8') as f:
#                 json.dump(corpus_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
#     return corpus_dict


# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            text = ''.join(text.splitlines())
            pdf_text += text
        else:
            # 使用 OCR 方式擷取頁面文字
            image = page.to_image(resolution=500)  # 提取頁面的文本畫面
            ocr_text = pytesseract.image_to_string(image.original, lang='chi_tra', config="--psm 12")
            if ocr_text:
                ocr_text = ''.join(ocr_text.replace(" ", "").splitlines())
                pdf_text += ocr_text

        # only use OCR
        # image = page.to_image(resolution=500)  # 提取頁面的文本畫面
        # ocr_text = pytesseract.image_to_string(image.original, lang='chi_tra', config="--psm 12")
        # if ocr_text:
        #     ocr_text = ''.join(ocr_text.replace(" ", "").splitlines())
        #     pdf_text += ocr_text

    pdf.close()  # 關閉PDF文件

    return pdf_text  # 返回萃取出的文本

def cut_words(corpus: str) -> list[str]:
    result = []
    corpus = corpus.replace(" ", "")
    stop_word = {}.fromkeys([line.strip() for line in open(rb'./reference/stop_word.txt', encoding='utf-8')])

    for word in jieba.lcut_for_search(corpus, HMM=False):
        if word != ' ' and word not in stop_word:
            result.append(word)
    return result

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

def text2vec_process(query: str, candidate_corpus: list[str]) -> str:
    best_index = 0
    best_similarity = 0
    chunk_size = 100
    overlap_size = 50

    query_embedding = embedder.encode(query)
    for index, candidate in enumerate(candidate_corpus):
        # if len(candidate) < 200:
        #     temp_chunks = [candidate]
        # else:
        #     temp_chunks = [candidate[i:i + chunk_size] for i in range(0, len(candidate), chunk_size - overlap_size)]
        modify_candidate = ''.join(candidate.replace(" ", "").splitlines())
        sentence = list(filter(None, re.split('[.。,，;；?？]', modify_candidate)))
        temp_chunks = sentence
        # temp_chunks = [''.join(sentence[i:i + 1]) for i in range(0, len(sentence), 1 - 0)]

        corpus_embeddings = embedder.encode(temp_chunks)
        hits = semantic_search(query_embedding, corpus_embeddings, top_k=1)
        current_mean_similarity_score = [hit["score"] for hit in hits[0]]
        current_similarity = fmean(current_mean_similarity_score)
        if current_similarity > best_similarity:
            best_index = index
            best_similarity = current_similarity

    if best_similarity < 0.9:
        best_index = 0
    # base on whole paragraph
    # corpus_embeddings = embedder.encode(candidate_corpus)
    # hits = semantic_search(query_embedding, corpus_embeddings, top_k=1)
    # best_index = hits[0][0]["corpus_id"]

    return candidate_corpus[best_index]

# 根據查詢語句和指定的來源，檢索答案
def BM25_retrieve(qs, source, corpus_dict):
    filtered_corpus = [corpus_dict[int(file)] for file in source]

    # [TODO] 可自行替換其他檢索方式，以提升效能

    tokenized_corpus = [cut_words(doc) for doc in filtered_corpus]  # 將每篇文檔進行分詞

    bm25 = BM25Plus(tokenized_corpus)  # 使用BM25演算法建立檢索模型
    tokenized_query = cut_words(qs)  # 將查詢語句進行分詞
    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=3)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
    a = text2vec_process(qs, ans)
    if ans[0] != a:
        pass
    # 找回與最佳匹配文本相對應的檔案名
    res = [key for key, value in corpus_dict.items() if value == a]
    return res[0]  # 回傳檔案名


if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

    args = parser.parse_args()  # 解析參數

    start_time = perf_counter()

    # 初始化套件資訊
    jieba.initialize()
    # 設定繁體中文主辭典
    jieba.set_dictionary('./reference/dict.txt')
    # 加入自定義辭典
    jieba.load_userdict("./reference/user_dict.txt")

    answer_dict = {"answers": []}  # 初始化字典

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    if os.path.isfile("corpus_dict/insurance_corpus_dict.json"):
        with open("corpus_dict/insurance_corpus_dict.json", 'rb') as f:
            corpus_dict_insurance =  {int(k): v for k, v in json.load(f).items()}
    else:
        print("No insurance_corpus_dict.json load insurance data")
        source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
        corpus_dict_insurance = load_data(source_path_insurance)
        with open("corpus_dict/insurance_corpus_dict.json", 'w', encoding='utf8') as f:
            json.dump(corpus_dict_insurance, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符

    if os.path.isfile("corpus_dict/finance_corpus_dict.json"):
        with open("corpus_dict/finance_corpus_dict.json", 'rb') as f:
            corpus_dict_finance = {int(k): v for k, v in json.load(f).items()}
    else:
        print("No finance_corpus_dict.json load finance data")
        source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
        corpus_dict_finance = load_data(source_path_finance)
        with open("corpus_dict/finance_corpus_dict.json", 'w', encoding='utf8') as f:
            json.dump(corpus_dict_finance, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符

    if os.path.isfile("corpus_dict/faq_corpus_dict.json"):
        with open("corpus_dict/faq_corpus_dict.json", 'rb') as f:
            key_to_source_dict = {int(k): v for k, v in json.load(f).items()}
    else:
        print("No faq_corpus_dict.json load faq data")
        with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
            key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
            key_to_source_dict = {int(key): read_faq_data(value) for key, value in key_to_source_dict.items()} # 整理讀取資料
        with open("corpus_dict/faq_corpus_dict.json", 'w', encoding='utf8') as f:
            json.dump(key_to_source_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符

    for q_dict in tqdm(qs_ref['questions']):
        if (len(answer_dict['answers']) >= 143):
            pass
        if q_dict['category'] == 'finance':
            # 進行檢索
            retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            corpus_dict_faq = {key: value for key, value in key_to_source_dict.items() if key in q_dict['source']}
            retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_faq)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符

    print(f"execution time: {perf_counter() - start_time:.2f}s")
