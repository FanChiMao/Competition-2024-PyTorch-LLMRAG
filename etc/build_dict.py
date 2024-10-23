import time
from selenium import webdriver
from selenium.webdriver.common.by import By

if __name__ == "__main__":
    dict_path = "data/fsc_dict.txt"
    entry = "https://www.fsc.gov.tw/ch/home.jsp?id=178&parentpath=0%2C6&mcustomize="
    driver = webdriver.Edge()
    driver.get(entry)

    keywords = set()

    while True:
        keyword_elements = driver.find_elements(By.XPATH, "//td[@data-th='中文詞彙']")
        current_page = set()
        for element in keyword_elements:
            current_page |= set([txt.replace(" ", "") + "\n" for txt in element.text.split("；")])
        
        if current_page.issubset(keywords):
            break

        keywords |= current_page

        next_page_button = driver.find_element(By.XPATH, "//a[@title='下一頁']")
        next_page_button.click()

        time.sleep(0.5)

    with open(dict_path, "w") as f:
        f.writelines(keywords)

    driver.quit()