import jieba_TW.jieba as jieba
from pathlib import Path

def cut_words(corpus: str) -> list[str]:
    result = []
    corpus = corpus.replace(" ", "")
    stop_word = {}.fromkeys([line.strip() for line in open(Path("./reference/stop_word.txt"), "r+", encoding='utf-8')])

    for word in jieba.lcut_for_search(corpus, HMM=False):
        if word != ' ' and word not in stop_word:
            result.append(word)
    return result


def main() -> None:

    # 加入自定義辭典
    jieba.load_userdict("./data/custom_words/user_dict.txt")

    print(cut_words("如何更新刷臉ID? (1) 持玉山銀行任一張有效之金融卡到玉山銀行營業部、城中分行廳內之刷臉ATM進行更新。(2) 在ATM螢幕點選「無卡交易」→「刷臉提款」→「更新刷臉ID」→插入「金融卡」→輸入「晶片金融卡密碼」→系統會傳送一組簡訊密碼至您的手機，請您查收簡訊並於ATM輸入「簡訊密碼」→設定「刷臉ID」即可完成變更。 更新刷臉ID需要哪些步驟? (1) 持玉山銀行任一張有效之金融卡到玉山銀行營業部、城中分行廳內之刷臉ATM進行更新。(2) 在ATM螢幕點選「無卡交易」→「刷臉提款」→「更新刷臉ID」→插入「金融卡」→輸入「晶片金融卡密碼」→系統會傳送一組簡訊密碼至您的手機，請您查收簡訊並於ATM輸入「簡訊密碼」→設定「刷臉ID」即可完成變更。 更新刷臉ID需要使用金融卡嗎? 是，需要持玉山銀行任一張有效之金融卡 更新刷臉ID過程中需要輸入哪些資訊? 金融卡密碼 簡訊密碼 "))

if __name__ == '__main__':
    main()