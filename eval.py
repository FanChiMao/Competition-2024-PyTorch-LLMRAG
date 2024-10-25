import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--groundtruth_path', type=str, default="./dataset/preliminary/ground_truths_example.json", help="path to ground truth json")  # 問題文件的路徑
    parser.add_argument('--submission_path', type=str, default="./dataset/preliminary/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/answer_Chroma_50_25.json", help="path to submission json")  # 參考資料的路徑

    args = parser.parse_args()

    with open(args.groundtruth_path, 'r') as f:
        ground_truth = json.load(f)  # 讀取問題檔案
        print("%s is loaded."%args.groundtruth_path)
    
    with open(args.submission_path, 'r') as f:
        submission = json.load(f)  # 讀取問題檔案
        print("%s is loaded."%args.submission_path)


    ground_truth = ground_truth["ground_truths"]
    submission = submission["answers"]
    # ground_truth = ground_truth["ground_truths"][:50]
    # submission = submission["answers"][:50]
    # ground_truth = ground_truth["ground_truths"][50:100]
    # submission = submission["answers"][50:100]
    # ground_truth = ground_truth["ground_truths"][100:]
    # submission = submission["answers"][100:]
    # 計算分數

    precision = 0
    precision50 = 0
    precision100 = 0
    precision150 = 0
    wrong_qids = []
    for sub, gt in zip(submission, ground_truth):
        if sub["qid"] != gt["qid"]:
            print("qid not match")
            break
        if sub["retrieve"] == gt["retrieve"]:
            precision += 1
        else:
            wrong_qids.append(str(sub["qid"]))
        if sub["qid"] == 50:
            precision50 = precision / 50
            precision = 0
        if sub["qid"] == 100: 
            precision100 = precision / 50
            precision = 0
        if sub["qid"] == 150: 
            precision150 = precision / 50
            precision = 0

    print("Wrong qid: %s"% ",".join(wrong_qids))
    print("Insurance Precision: {:.2f} %".format(precision50 * 100))
    print("Finance Precision: {:.2f} %".format(precision100 * 100))
    print("FAQ Precision: {:.2f} %".format(precision150 * 100))

    avg_precision = (precision50 + precision100 + precision150)/3
    print("Avg Precision: {:.2f} %".format(avg_precision * 100))