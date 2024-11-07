import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, default="./data/datasets/preliminary/questions_example_revision.json", help="path to question json")  # 問題文件的路徑
    parser.add_argument('--groundtruth_path', type=str, default="./data/datasets/preliminary/ground_truths_example_revision.json", help="path to ground truth json")  # 問題文件的路徑
    parser.add_argument('--submission_path', type=str, default="outputs/preliminary/submission_2.json", help="path to submission json")  # 參考資料的路徑

    args = parser.parse_args()

    with open(args.question_path, 'r', encoding='utf-8') as f:
        question = json.load(f)  # 讀取問題檔案

    with open(args.groundtruth_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)  # 讀取問題檔案
    
    with open(args.submission_path, 'r', encoding='utf-8') as f:
        submission = json.load(f)  # 讀取問題檔案

    q = question["questions"]
    # 計算分數
    precision = 0
    n_wrong = 0
    precision_insurance = 0
    precision_finance = 0
    precision_faq = 0
    insurance_count = 0
    finance_count = 0
    faq_count = 0

    n_questions = len(ground_truth["ground_truths"])
    wrong_qid = []
    print("錯誤題目：")
    print("=" * 100)
    for sub, gt in zip(submission["answers"], ground_truth["ground_truths"]):
        if sub["qid"] != gt["qid"]:
            assert ValueError(f"qid not match: {sub['qid']} != {gt['qid']}")

        if isinstance(sub["retrieve"], list):  # get top 1 if it is a list (top n)
            sub["retrieve"] = sub["retrieve"][0]

        if 1 <= gt["qid"] <= 50:
            if sub["retrieve"] == gt["retrieve"]:
                precision_insurance += 1
            insurance_count += 1

        elif 51 <= gt["qid"] <= 100:
            if sub["retrieve"] == gt["retrieve"]:
                precision_finance += 1
            finance_count += 1

        elif 101 <= gt["qid"] <= 150:
            if sub["retrieve"] == gt["retrieve"]:
                precision_faq += 1
            faq_count += 1


        if sub["retrieve"] == gt["retrieve"]:
            precision += 1
        else:
            n_wrong += 1
            wrong_qid.append(gt["qid"])
            print("qid {}: {} ans: {}, gt: {}".format(gt["qid"], q[sub["qid"]-1]["query"], sub["retrieve"], gt["retrieve"]))

        if gt["qid"] == 50 or gt["qid"] == 100 or gt["qid"] == 150:
            print("=" * 100)

    precision /= n_questions
    print(f"Wrong id: " + ", ".join([str(item) for item in wrong_qid]))
    print("Total questions: {}, Wrong answers: {}".format(n_questions, n_wrong))
    print(f"Insurance precision: {precision_insurance/50 * 100:.2f} %; ({precision_insurance}/50)")
    print(f"Finance precision:   {precision_finance/50 * 100:.2f} %; ({precision_finance}/50)")
    print(f"FAQ precision:       {precision_faq/50 * 100:.2f} %; ({precision_faq}/50)")
    print(f"Total Precision:     {precision * 100:.2f} %: ({n_questions - n_wrong}/{n_questions})")
