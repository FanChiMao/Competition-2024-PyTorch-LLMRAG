import os
import re
import pdfplumber
from tqdm import tqdm
from glob import glob


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


def extract_text_within_brackets(text):
    pattern = r'「(.*?)」'  # Regex pattern to match text within 「」
    return re.findall(pattern, text)


def print_corpus_in_brackets(pdf_folder_path):
    printed_words = set()
    pdf_paths = glob(os.path.join(pdf_folder_path, "*.pdf"))
    for _, pdf_path in enumerate(pdf_paths):
        content_str = read_pdf(pdf_path)
        certain_words = extract_text_within_brackets(content_str)
        for word in certain_words:
            if word not in printed_words:
                print(f"{os.path.basename(pdf_path)}: \"{word}\"")
                printed_words.add(word)

    return list(printed_words)


def save_corpus(corpus, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in corpus:
            f.write(f"{line}\n")


if __name__ == '__main__':
    # Print to check the parsing result
    # content = read_pdf(r"D:\Others\LLM_RAG\Competition-2024-PyTorch-LLMRAG-main\datasets\preliminary\finance\76.pdf")
    # print(content)


    # print the words in 「」
    corpus = print_corpus_in_brackets(r"D:\Others\LLM_RAG\Competition-2024-PyTorch-LLMRAG-main\datasets\preliminary\finance")
    save_corpus(corpus, "finance.txt")

