import re
import cn2an
import string

def season_convert(text):
    pattern = r"第[一二三四1234]季"
    matched = re.search(pattern, text)
    if matched:
        season_str = matched.group(0)
        season = season_str.replace("第", "").replace("季", "")
        season = cn2an.cn2an(season, "smart")
        
        season_months = ""
        if season == 1:
            season_months = "1月2月3月"
        elif season == 2:
            season_months = "4月5月6月"
        elif season == 3:
            season_months = "7月8月9月"
        elif season == 4:
            season_months = "10月11月12月"

        replace_date = "{}/{}".format(season_str, season_months)
        text = re.sub(pattern, replace_date, text)
    return text

def age_convert(text):
    pattern = r"[一二三四五六七八九零十百○]+(歲|足歲)"
    matched = re.search(pattern, text)
    if matched:
        age_str = matched.group(0)
        age = age_str.replace("足歲", "").replace("歲", "").replace("○", "零")
        age = str(cn2an.cn2an(age, "smart"))
        
        replace_date = "{}/{}".format(age, age_str)
        text = re.sub(pattern, replace_date, text)
    return text

def ROCC_convert(text):    
    pattern = r"(?<!\d)\d{2,3}(?!\d)年"
    matched = re.search(pattern, text)
    if matched:
        date_str = matched.group(0)
        year = date_str.replace("年", "")
        year = int(year)+1911
        replace_date = "{}/{}年".format(date_str, year)
        text = re.sub(pattern, replace_date, text)

    pattern = r"民國[一二三四五六七八九零十百○]+年[一二三四五六七八九零十百○]+月[一二三四五六七八九零十百○]+日"
    matched = re.search(pattern, text)
    if matched:
        date_str = matched.group(0)
        
        m_year = re.search("民國(.*)年", date_str)
        year = m_year.group(0).replace("民國", "").replace("年", "").replace("○", "零")
        year = cn2an.cn2an(year, "smart")
        year = int(year)+1911

        m_month = re.search("年(.*)月", date_str)
        month = m_month.group(0).replace("年", "").replace("月", "")
        month = cn2an.cn2an(month, "smart")

        m_day = re.search("月(.*)日", date_str)
        day = m_day.group(0).replace("月", "").replace("日", "")
        day = cn2an.cn2an(day, "smart")

        replace_date = "{}/{}年{}月{}日".format(date_str, year, month, day)
        text = re.sub(pattern, replace_date, text)
    return text

def remove_punctuation(text):
    punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.\n\/ /"
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub("[{}]+".format(punc), "", text)
    return text

def kelvin_preprocess(text):
    # [Kelvin] 清除特殊符號
    text = remove_punctuation(text)
    # [Kelvin] 民國年轉換
    text = ROCC_convert(text)
    # [Kelvin] 季度轉換
    #text = season_convert(text)
    # [Kelvin] 歲數轉換
    #text = age_convert(text)
    return text


########################################################################################################################
def read_faq_data(faq_items: list) -> str:
    text = ""
    for item in faq_items:
        if 'question' in item:
            text += item['question'] + " "
        if 'answers' in item:
            for temp_answer in item['answers']:
                text += temp_answer + " "
    text = ''.join(text.splitlines())
    return text