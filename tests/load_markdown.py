import re


if __name__ == '__main__':
    path = r"F:\finance\1000\1000.md"

    with open(path, "r", encoding="utf-8") as file:
        markdown_text = file.read()

    markdown_text = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_text)  # ignore image tag
    markdown_text = markdown_text.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "").replace("|", "").replace("-", "")
    print(markdown_text)
