import os
import json
import argparse
from collections import defaultdict

from src.pipeline.pipelines import KelvinPipeline, JonathanPipeline, TomPipeline


def RRF(*ranked_lists, k=60):
    """
    Perform Reciprocal Rank Fusion (RRF) on the provided ranked lists.
    Each item in ranked_lists is a dictionary with 'id' as the document identifier.
    """
    rrf_scores = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, predict_id in enumerate(ranked_list):
            doc_id = predict_id
            rrf_scores[doc_id] += 1 / (k + rank + 1)  # Reciprocal rank calculation

    # Sort by RRF score (higher is better) and return the top_n items
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, default="./data/datasets/preliminary/questions_example_revision.json", help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, default="./data/datasets/preliminary", help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, default="./outputs/submission_RRF_k_3.json", help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--top_n', type=int, default=3, help='選擇TOPN')
    parser.add_argument('--save_each_result', type=bool, default=True, help='是否除最終 json 結果之外，同時將各個 pipeline 的 json 結果也儲存')
    parser.add_argument('--yaml', type=str, default='./data/pipeline.yml', help='pipeline專用參數')  # 選擇模型
    args = parser.parse_args()  # 解析參數

    answer_dict_0 = KelvinPipeline(args).run()
    answer_dict_1 = JonathanPipeline(args).run()
    answer_dict_2 = TomPipeline(args).run()

    if args.save_each_result:
        with open(args.output_path.replace(".json", "_0.json"), 'w', encoding='utf8') as f:
            json.dump(answer_dict_0, f, ensure_ascii=False, indent=4)
        with open(args.output_path.replace(".json", "_1.json"), 'w', encoding='utf8') as f:
            json.dump(answer_dict_1, f, ensure_ascii=False, indent=4)
        with open(args.output_path.replace(".json", "_2.json"), 'w', encoding='utf8') as f:
            json.dump(answer_dict_2, f, ensure_ascii=False, indent=4)


    final_answers = {"answers": []}

    for answer_0, answer_1, answer_2 in zip(answer_dict_0["answers"], answer_dict_1["answers"], answer_dict_2["answers"]):
        top_n_lists = [answer_0["retrieve"], answer_1["retrieve"], answer_2["retrieve"]]
        fused_list = RRF(*top_n_lists, k=3)[:args.top_n]
        fused_answer = {"qid": answer_0['qid'], "retrieve": fused_list[0][0]}
        final_answers["answers"].append(fused_answer)

    # 將答案字典保存為json文件
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(final_answers, f, ensure_ascii=False, indent=4)
