import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--groundtruth_path', type=str, default="./data/dataset/preliminary/ground_truths_example.json", help="path to ground truth json")  # 問題文件的路徑
    parser.add_argument('--submission_path', type=str, default="./output/baseline.json", help="path to submission json")  # 參考資料的路徑

    args = parser.parse_args()

    with open(args.groundtruth_path, 'r') as f:
        ground_truth = json.load(f)  # 讀取問題檔案
    
    with open(args.submission_path, 'r') as f:
        submission = json.load(f)  # 讀取問題檔案

    # 計算分數
    precision = 0
    n_questions = len(ground_truth["ground_truths"])

    # 計算題目類型分數
    category_precision = {
        "faq": [0, 0],
        "finance": [0, 0],
        "insurance": [0, 0],
    }

    wrong_answers = {
        "insurance": [],
        "finance": [],
        "faq": []
    }

    for sub, gt in zip(submission["answers"], ground_truth["ground_truths"]):
        if sub["qid"] != gt["qid"]:
            print("qid not match")
            break
        if sub["retrieve"] == gt["retrieve"]:
            category_precision[gt["category"]][0] += 1
            precision += 1
            # print(gt["qid"], gt["category"], sub["retrieve"])
        else:
            # print(gt["qid"], gt["category"], sub["retrieve"])
            wrong_answers[gt["category"]].append({gt["qid"]: sub["retrieve"]})

        category_precision[gt["category"]][1] += 1

    precision /= n_questions

    with open("wrong_answers_dict.json", 'w', encoding='utf8') as f:
        json.dump(wrong_answers, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符

    print("Precision: {:.2f} % [faq: {:.2f} %, finance: {:.2f} %, insurance: {:.2f} %]".format(precision * 100, category_precision["faq"][0] / category_precision["faq"][1] * 100, category_precision["finance"][0] / category_precision["finance"][1] * 100, category_precision["insurance"][0] / category_precision["insurance"][1] * 100))
    print(category_precision)