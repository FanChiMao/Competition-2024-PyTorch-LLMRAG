import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--groundtruth_path', type=str, default="./datasets/preliminary/ground_truths_example.json")  # 問題文件的路徑
    parser.add_argument('--submission_path', type=str, default="./output/baseline.json")  # 參考資料的路徑

    args = parser.parse_args()

    with open(args.groundtruth_path, 'r') as f:
        ground_truth = json.load(f)  # 讀取問題檔案

    with open(args.submission_path, 'r') as f:
        submission = json.load(f)  # 讀取問題檔案

    # 計算分數
    precision = 0
    n_questions = len(ground_truth["ground_truths"])
    for sub, gt in zip(submission["answers"], ground_truth["ground_truths"]):
        if sub["qid"] != gt["qid"]:
            print("qid not match")
            break
        if sub["retrieve"] == gt["retrieve"]:
            precision += 1
    precision /= n_questions

    print("Precision: {:.2f} %".format(precision * 100))