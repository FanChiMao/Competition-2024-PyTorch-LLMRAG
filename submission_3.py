"""
Preliminary Submission 3
========================================================================================================================
Only Use Edward pipelines (Based on embedding similarity)

Use embedding model (BAAI/bge-small-zh-v1.5) to retrieval the top-3 passages for Edward.
Following is the simple graphics of the flowchart

                                    +--------------------+
                                    |   Edward Pipeline  |
                                    +--------------------+
                                              |
                                              v
                                +-----------------------------+
                                |      Embedding Model        |
                                +-----------------------------+
                                              |
                                              v
                                +-----------------------------+
                                |   Retrieve Top-3 Passages   |
                                +-----------------------------+
                                              |
                                              v
                                +-----------------------------+
                                |     Final Top-1 Results     |
                                +-----------------------------+

========================================================================================================================
"""
import os
import json
import argparse

from src.retrieve.reranker import RRF
from src.pipeline.pipelines import KelvinPipeline, JonathanPipeline, TomPipeline, EdwardPipeline


if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, default="./data/datasets/preliminary_questions/questions_preliminary.json", help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, default="./data/datasets/preliminary", help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, default="./outputs/preliminary/submission_3.json", help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--top_n', type=int, default=3, help='選擇TOPN')
    parser.add_argument('--save_each_result', type=bool, default=True, help='是否除最終 json 結果之外，同時將各個 pipeline 的 json top_n 結果也儲存')
    parser.add_argument('--yaml', type=str, default='./data/pipeline.yml', help='pipeline專用參數')  # 選擇模型
    args = parser.parse_args()  # 解析參數

    # inference with each pipeline
    answer_dict_edward = EdwardPipeline(args).run()
    if args.save_each_result:
        with open(args.output_path.replace(f".json", f"_edward_top{args.top_n}.json"), 'w', encoding='utf8') as f:
            json.dump(answer_dict_edward, f, ensure_ascii=False, indent=4)

    final_answers = {"answers": []}
    for a_edward in answer_dict_edward["answers"]:
        fused_answer = {"qid": a_edward['qid'], "retrieve": a_edward["retrieve"][0]}  # choose top 1
        final_answers["answers"].append(fused_answer)

    # 將答案字典保存為json文件
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(final_answers, f, ensure_ascii=False, indent=4)
