import os
import csv
from tqdm import tqdm
from glob import glob
import statistics
from PyPDF2 import PdfReader

from pdf2image import convert_from_path
import re
import fitz

def count_words_in_pdf(pdf_path):
    word_count = 0
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        words = text.split()
        word_count = len(words)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return word_count

def count_images_in_pdf(pdf_data_path):
    try:
        image_list = convert_from_path(pdf_data_path)
        return len(image_list)
    except:
        return 0


if __name__ == '__main__':
    dataset_dir = r"D:\Others\LLM_RAG\Competition-2024-PyTorch-LLMRAG-main\datasets\preliminary"

    pdf_data_list = []
    word_counts = []
    image_counts = []

    pdf_files = sorted(glob(os.path.join(dataset_dir, "*", "*.pdf")))
    for _, pdf_path in enumerate(tqdm(pdf_files)):
        # print("=> pdf_file: ", pdf_path)

        word_count = count_words_in_pdf(pdf_path)
        image_count = count_images_in_pdf(pdf_path)

        pdf_data_list.append({
            "File Name": pdf_path,
            "Word Count": word_count,
            "Image Count": image_count
        })

        word_counts.append(word_count)
        image_counts.append(image_count)

    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Median word count: {statistics.median(word_counts)}")
    print(f"Median image count: {statistics.median(image_counts)}")
    print(f"Average word count: {statistics.mean(word_counts)}")
    print(f"Average image count: {statistics.mean(image_counts)}")

    csv_file_path = "./data_analyze_results.csv"
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["File Name", "Word Count", "Image Count"])
        writer.writeheader()
        writer.writerows(pdf_data_list)