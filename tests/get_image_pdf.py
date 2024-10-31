import os
import pdfplumber
from glob import glob
from tqdm import tqdm

if __name__ == '__main__':
    root = f"../data/datasets/preliminary"

    pdf_paths = glob(os.path.join(root, "*", "*.pdf"))

    contain_image_pdf = []
    for i, pdf_path in enumerate(tqdm(pdf_paths)):
        pdf = pdfplumber.open(pdf_path)

        pages = pdf.pages
        pdf_text = ''
        for i, page in enumerate(pages):  # 迴圈遍歷每一頁
            text = page.extract_text()
            if text == "":
                contain_image_pdf.append(pdf_path)
                print(f"Page {i+1}:" + pdf_path)

    print(contain_image_pdf)
