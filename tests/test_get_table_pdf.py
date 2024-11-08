import os
import pdfplumber
from glob import glob
from tqdm import tqdm
from data.additional_info.info import finance_image_id_list, finance_table_id_list

# Initialize the lists
messy_table_id_list = []
messy_table_page_list = []

if __name__ == '__main__':
    from data.additional_info.info import messy_table_id_list
    root = f"../data/datasets/preliminary/finance"
    pdf_paths = glob(os.path.join(root, "*.pdf"))
    have_processed_id_list = finance_image_id_list + finance_table_id_list

    contain_table_pdf = []
    for pdf_path in tqdm(pdf_paths):
        pdf = pdfplumber.open(pdf_path)
        file_id = int(os.path.basename(pdf_path).split(".")[0])

        if file_id in have_processed_id_list:
            continue

        pages = pdf.pages
        pdf_text = ''
        found_table_pages = []
        for i, page in enumerate(pages):  # Loop through each page
            text = page.extract_text()
            if text and text.count("\n") > 100:
                # A table is likely present
                contain_table_pdf.append(pdf_path)
                found_table_pages.append(i + 1)  # Save page number (1-indexed)
                print(f"Page {i+1}: " + pdf_path)

        if found_table_pages:  # If any pages with potential tables were found
            messy_table_id_list.append(file_id)
            messy_table_page_list.append(found_table_pages)

    print("messy_table_id_list =", messy_table_id_list)
    print("messy_table_page_list =", messy_table_page_list)
