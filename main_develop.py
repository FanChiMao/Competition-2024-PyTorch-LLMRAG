import os
import json
import argparse

from tqdm import tqdm
import jieba
import jieba_TW.jieba as jieba_tw
import pdfplumber
from rank_bm25 import BM25Okapi, BM25Plus
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util


def load_data(source_path):
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in
                   tqdm(masked_file_ls)}  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    return corpus_dict


def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text
    pdf.close()  # 關閉PDF文件

    return pdf_text  # 返回萃取出的文本


def load_corpus_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_corpus_dict(corpus_dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(corpus_dict, f)


def retriever(qs, source, corpus_dict, tokenization='ch', reranker=False):
    if tokenization == 'ch':
        tokenizer = jieba
    elif tokenization == 'tw':
        tokenizer = jieba_tw
        tokenizer.dt.cache_file = 'jieba.cache.tw'
    else:
        raise ValueError(f"Invalid tokenization method \"{tokenization}\"")

    filtered_corpus = [corpus_dict[int(file)] for file in source]
    tokenized_corpus = [list(tokenizer.cut_for_search(doc)) for doc in filtered_corpus]
    bm25 = BM25Plus(tokenized_corpus)
    tokenized_query = list(tokenizer.cut_for_search(qs))
    top_docs = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=5)

    if reranker:
        reranker_model = SentenceTransformer('BAAI/bge-m3')  # bge-base-zh-v1.5 (small / base / large) / bge-m3
        # reranked_docs = rerank_with_sentence_transformer(qs, top_docs, reranker_model)
        reranked_docs = rerank_with_chunking(qs, top_docs, reranker_model)
        a = reranked_docs[0]
    else:
        a = top_docs[0]

    res = [key for key, value in corpus_dict.items() if value == a]  # 找回與最佳匹配文本相對應的檔案名
    return res[0]  # 回傳檔案名



def chunk_docs_string(text, embedder_model, max_tokens=512):
    chunks = []
    for i in range(0, len(text), max_tokens):
        chunk_text = text[i:i + max_tokens]
        chunks.append(chunk_text)
    return chunks


def rerank_with_chunking(query, top_docs, embedder_model, max_tokens=512):
    query_embedding = embedder_model.encode(query, normalize_embeddings=True, convert_to_tensor=True)

    doc_to_max_similarity = []
    for doc in top_docs:
        chunk_texts = chunk_docs_string(doc, embedder_model, max_tokens=max_tokens)
        chunk_embeddings = embedder_model.encode(chunk_texts, normalize_embeddings=True, convert_to_tensor=True)
        chunk_similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)

        max_similarity = np.max(chunk_similarities.cpu().numpy())
        doc_to_max_similarity.append(max_similarity)


    ranked_docs = sorted(zip(top_docs, doc_to_max_similarity), key=lambda x: x[1], reverse=True)

    return [doc[0] for doc in ranked_docs]


def rerank_with_sentence_transformer(query, top_docs, embedder_model: SentenceTransformer = None):
    if embedder_model is None:
        return top_docs

    # instruction = "为这个句子生成表示以用于检索相关文章："  [instruction + q for q in query]

    query_embedding = embedder_model.encode(query, normalize_embeddings=True, convert_to_tensor=True)
    doc_embeddings = embedder_model.encode(top_docs, normalize_embeddings=True, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)
    similarity_scores = similarities[0].tolist()

    # 將文檔按照相似度重新排序
    ranked_docs = sorted(zip(top_docs, similarity_scores), key=lambda x: x[1], reverse=True)
    return [doc[0] for doc in ranked_docs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, default=r".\datasets\preliminary\questions_example.json")
    parser.add_argument('--source_path', type=str, default=r".\datasets\preliminary")
    parser.add_argument('--tokenization', type=str, default="tw", help='結巴斷詞')  # tw, ch
    parser.add_argument('--reranker', type=bool, default=False, help='是否使用 Re-ranker')
    parser.add_argument('--output_path', type=str, default="./result_top5.json")
    args = parser.parse_args()

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)

    ####################################################################################################################
    # Load or create the corpus dict for each category
    if os.path.exists("./insurance_corpus.pkl"):
        corpus_dict_insurance = load_corpus_dict("./insurance_corpus.pkl")
    else:
        source_path_insurance = os.path.join(args.source_path, 'insurance')
        corpus_dict_insurance = load_data(source_path_insurance)
        save_corpus_dict(corpus_dict_insurance, "./insurance_corpus.pkl")

    if os.path.exists("./finance_corpus.pkl"):
        corpus_dict_finance = load_corpus_dict("./finance_corpus.pkl")
    else:
        source_path_finance = os.path.join(args.source_path, 'finance')
        corpus_dict_finance = load_data(source_path_finance)
        save_corpus_dict(corpus_dict_finance, "./finance_corpus.pkl")

    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    ####################################################################################################################
    # Star answering the questions
    answer_dict = {"answers": []}
    for _, q_dict in enumerate(tqdm(qs_ref['questions'], desc="Answering the questions")):
        if q_dict['category'] == 'finance':
            corpus_dict = corpus_dict_finance
        elif q_dict['category'] == 'insurance':
            corpus_dict = corpus_dict_insurance
        elif q_dict['category'] == 'faq':
            corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
            corpus_dict = corpus_dict_faq
        else:
            raise ValueError(f"Category must to be \"finance\", \"insurance\" or \"faq\", get \"{q_dict['category']}\" instead.")

        retrieved = retriever(q_dict['query'], q_dict['source'], corpus_dict, args.tokenization, args.reranker)
        answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

    ####################################################################################################################
    # Save the results
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
