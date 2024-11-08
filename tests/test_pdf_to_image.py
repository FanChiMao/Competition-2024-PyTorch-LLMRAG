import fitz  # PyMuPDF
import os
from tqdm import tqdm

from data.additional_info.info import messy_table_id_list, messy_table_page_list


def convert_pdf_pages_to_images(pdf_path, output_folder, page_numbers):
    file_id = int(os.path.basename(pdf_path).split(".")[0])
    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        # Convert only specified pages
        if (i + 1) in page_numbers:  # Page numbers are assumed to be 1-indexed
            page = doc.load_page(i)  # Load page
            pix = page.get_pixmap()  # Render page to an image
            output_path = f"{output_folder}/{file_id}_{i + 1}.png"
            pix.save(output_path)
            print(f"Saved: {output_path}")


if __name__ == '__main__':
    from glob import glob

    finance_path = r"F:\Competition-2024-PyTorch-LLMRAG\data\datasets\preliminary\finance"
    save_path = r"F:\Competition-2024-PyTorch-LLMRAG\data\datasets\preliminary\finance_messy_table_images"

    pdf_files = glob(os.path.join(finance_path, "*.pdf"))
    for pdf_path in tqdm(pdf_files):
        file_id = int(os.path.basename(pdf_path).split(".")[0])

        if file_id in messy_table_id_list:
            # Find the index of the file_id to get corresponding page numbers
            index = messy_table_id_list.index(file_id)
            page_numbers = messy_table_page_list[index]  # Corresponding page numbers
            convert_pdf_pages_to_images(pdf_path, save_path, page_numbers)
