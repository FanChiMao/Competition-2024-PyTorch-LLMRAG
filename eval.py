import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, default="./data/dataset/preliminary/questions_example_revision.json", help="path to question json")  # 問題文件的路徑
    parser.add_argument('--groundtruth_path', type=str, default="./data/dataset/preliminary/ground_truths_example_revision.json", help="path to ground truth json")  # 問題文件的路徑
    parser.add_argument('--submission_path', type=str, default="./output/submission.json", help="path to submission json")  # 參考資料的路徑

    args = parser.parse_args()

    with open(args.question_path, 'r') as f:
        question = json.load(f)  # 讀取問題檔案

    with open(args.groundtruth_path, 'r') as f:
        ground_truth = json.load(f)  # 讀取問題檔案
    
    with open(args.submission_path, 'r') as f:
        submission = json.load(f)  # 讀取問題檔案

    q = question["questions"]
    # 計算分數
    precision = 0
    n_wrong = 0
    n_questions = len(ground_truth["ground_truths"])
    print("錯誤題目：")
    for sub, gt in zip(submission["answers"], ground_truth["ground_truths"]):
        if sub["qid"] != gt["qid"]:
            print("qid not match")
            break
        if sub["retrieve"] == gt["retrieve"]:
            precision += 1
        else:
            n_wrong += 1
            print("{}: ans: {}, gt: {}, category: {}".format(q[sub["qid"]-1]["query"], sub["retrieve"], gt["retrieve"], gt["category"]))
    precision /= n_questions
    print("="*100)
    print("Total questions: {}, Wrong answers: {}".format(n_questions, n_wrong))
    print("Precision: {:.2f} %".format(precision * 100))