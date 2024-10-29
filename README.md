# Install
## install python packages
```commandline
pip install -r ./requirements.txt
```

## install tesseract
- [tesseract installation](https://tesseract-ocr.github.io/tessdoc/Installation.html)
- 需要新增系統變數 `編輯系統環境變數 > 環境變數 > 系統變數 > PATH > 加入 C:\Program Files\Tesseract-OCR`

# File Description
- `corpus_dict`: 儲存擷取出來的文件文字
- `finance_corpus_ocr_compare`: 儲存全部使用OCR擷取、部分使用OCR擷取以及完全沒有使用OCR擷取的finanace文件結果
- `reference`: 原始參考文件以及jieba斷詞辭典
  - `combine_user_dict.txt`: 儲存多人整理的辭典
  - `dict.txt`: 繁體中文主辭典
  - `dict.txt.big`: jieba斷詞大數量主辭典
  - `stop_word.txt`: 停用詞詞表
  - `user_dict.txt`: 自定義詞典(主要針對參考文件內容處理)
- `word_freq`: 統計斷詞詞頻率結果

# Result
## Top n Compare
| n | select index | Precision(%) | faq score(%)  | finance score(%) | insurance score(%) |
|---|:------------:|--------------|---------------|------------------|--------------------|
| 1 |      0       | 85.33        | 98.00 (49/50) | 68.00 (34/50)    | 90.00 (45/50)      |
| 2 |      1       | 8.00         | 0.00 (0/50)   | 20.00 (10/50)    | 4.00 (2/50)        |
| 3 |      -1      | 4.67         | 2.00 (1/50)   | 10.00 (5/50)     | 2.00 (1/50)        |

`備註: 已檢查正確題號為重複; 不是所有題目都有超過3個選項，故n取3時index選擇-1;`

## Method Compare
| Method                    | Precision(%) | faq score(%)  | finance score(%) | insurance score(%) |
|---------------------------|--------------|---------------|------------------|--------------------|
| Current Best              | 85.33        | 98.00 (49/50) | 68.00 (34/50)    | 90.00 (45/50)      |
| without OCR               | 85.33        | 98.00 (49/50) | 68.00 (34/50)    | 90.00 (45/50)      |
| with full OCR for finance | 84.67        | 98.00 (49/50) | 66.00 (33/50)    | 90.00 (45/50)      |
| (New) Baseline            | 75.33        | 94.00(47/50)  | 52.00(26/50)     | 80.00(40/50)       |
| -                         | -            | -             | -                | -                  |
| (OLD) BM25+               | 74.00        | 88.00 (44/50) | 48.00 (24/50)    | 86.00 (43/50)      |
| (OLD) BM25L               | 63.33        | 80.00 (40/50) | 32.00(16/50)     | 78.00 (39/50)      |
| (OLD) Baseline            | 71.33        | 90.00 (45/50) | 44.00 (22/50)    | 80.00 (40/50)      |
