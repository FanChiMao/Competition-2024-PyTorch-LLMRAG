import os
import zipfile
import gdown

if __name__ == '__main__':
    drive_url = 'https://drive.google.com/file/d/1gM344J7oChX69f4LteDejLDUekwq_PXX/view?usp=sharing'
    gdown.download(drive_url, "./preliminary.zip", fuzzy=True)

    try:
        if os.path.exists("./preliminary.zip"):
            with zipfile.ZipFile("preliminary.zip") as zip_ref:
                zip_ref.extractall(".")

    finally:
        if os.path.exists("./preliminary.zip"):
            os.remove("./preliminary.zip")





