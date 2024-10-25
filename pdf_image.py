import pdfplumber
import pytesseract

if __name__ == '__main__':
    file_path = "reference/finance/1.pdf"

    pdf = pdfplumber.open(file_path)

    pages = pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()
        tables = page.extract_tables()
        # image = page.to_image(resolution=500)  # 提取頁面的文本畫面
        # text = pytesseract.image_to_string(image.original, lang='chi_tra', config="--psm 12")
        if text:
            text = ''.join(text.replace(" ", "").splitlines())
            pdf_text += text
    pdf.close()  # 關閉PDF文件

    print(pdf_text)


