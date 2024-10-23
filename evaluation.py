import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--gt', type=str, default="./datasets/preliminary/ground_truths_example_revision.json")  # 問題文件的路徑
    parser.add_argument('--rs', type=str, default=r"./result.json")  # 參考資料的路徑  71.33%

    args = parser.parse_args()

    with open(args.gt, 'r') as f:
        ground_truth = json.load(f)  # 讀取問題檔案

    with open(args.rs, 'r') as f:
        submission = json.load(f)  # 讀取問題檔案

    # 計算分數
    wrong_qid = []
    precision = 0
    n_questions = len(ground_truth["ground_truths"])
    for sub, gt in zip(submission["answers"], ground_truth["ground_truths"]):
        if sub["qid"] != gt["qid"]:
            print("qid not match")
            break
        if sub["retrieve"] == gt["retrieve"]:
            precision += 1
        else:
            wrong_qid.append(gt["qid"])

    print("Precision: {:.2f} % ".format(precision/n_questions * 100) + f"({precision}/{n_questions})")
    print("Wrong QID: " + ", ".join([str(item) for item in wrong_qid]))
