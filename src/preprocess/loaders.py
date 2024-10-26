import os.path

import pdfplumber
import pytesseract
from src.preprocess.base import BasePDFLoader
from src.preprocess.text_process import kelvin_preprocess


class KelvinPDFLoader(BasePDFLoader):
    def __init__(self, source_dir, pickle_path=None, n_jobs=16):
        super().__init__(source_dir, pickle_path, n_jobs)
        if pickle_path is not None and not os.path.exists(os.path.dirname(pickle_path)):
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    
    @staticmethod
    def read_pdf(pdf_loc, page_infos: list = None):
        pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件
        
        # [TODO]: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理
        # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
        pdf_text = ''
        for _, page in enumerate(pages):  # 迴圈遍歷每一頁
            text = page.extract_text()  # 提取頁面的文本內容
            
            if text:
                text = kelvin_preprocess(text)
                pdf_text += text        

        pdf.close()  # 關閉PDF文件

        return pdf_text  # 返回萃取出的文本


class JonathanPDFLoader(BasePDFLoader):
    def __init__(self, source_dir, pickle_path=None, n_jobs=16):
        super().__init__(source_dir, pickle_path, n_jobs)
        if pickle_path is not None and not os.path.exists(os.path.dirname(pickle_path)):
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

    @staticmethod
    def read_pdf(pdf_loc, page_infos: list = None):
        pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
        pdf_text = ''
        for _, page in enumerate(pages):  # 迴圈遍歷每一頁
            text = page.extract_text()  # 提取頁面的文本內容

            if text:
                pdf_text += text

        pdf.close()  # 關閉PDF文件

        pdf_text = pdf_text.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")

        return pdf_text  # 返回萃取出的文本


class TomPDFLoader(BasePDFLoader):
    def __init__(self, source_dir, pickle_path=None, n_jobs=16):
        super().__init__(source_dir, pickle_path, n_jobs)
        if pickle_path is not None and not os.path.exists(os.path.dirname(pickle_path)):
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

    @staticmethod
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
            # 20241025 Jonathan OCR 失敗 先省略 😢
            # else:
            #     # 使用 OCR 方式擷取頁面文字
            #     image = page.to_image(resolution=500)  # 提取頁面的文本畫面
            #     ocr_text = pytesseract.image_to_string(image.original, lang='chi_tra', config="--psm 12")
            #     if ocr_text:
            #         ocr_text = ''.join(ocr_text.replace(" ", "").splitlines())
            #         pdf_text += ocr_text

            # only use OCR
            # image = page.to_image(resolution=500)  # 提取頁面的文本畫面
            # ocr_text = pytesseract.image_to_string(image.original, lang='chi_tra', config="--psm 12")
            # if ocr_text:
            #     ocr_text = ''.join(ocr_text.replace(" ", "").splitlines())
            #     pdf_text += ocr_text

        pdf.close()  # 關閉PDF文件

        return pdf_text  # 返回萃取出的文本
