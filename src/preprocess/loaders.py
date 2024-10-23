import pdfplumber
from src.preprocess.base import BasePDFLoader
from src.preprocess.text_process import kelvin_preprocess

class KelvinPDFLoader(BasePDFLoader):
    def __init__(self, source_dir, pickle_path=None, n_jobs=16):
        super().__init__(source_dir, pickle_path, n_jobs)
    
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