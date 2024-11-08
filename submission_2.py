"""
Preliminary Submission 2
========================================================================================================================
Combination of Kelvin, Jonathan and Tom pipelines (All are based on BM25+)

Use BM25+ with embedding reranker (BAAI/bge-small-zh-v1.5) to get the top-3 documents for Kelvin, Jonathan, and Tom,
Finally, use RRF method to fuse the top-3 documents to get the final top-1 document.
Following is the simple graphics of the flowchart

+-------------------+   +-------------------+   +-------------------+
|  Kelvin Pipeline  |   | Jonathan Pipeline |   |   Tom Pipeline    |
+-------------------+   +-------------------+   +-------------------+
                  \               |               /
                   \              |              /
                    \             |             /
                     \            |            /
                      \           v           /
                       +---------------------+
                       | BM25+ with Reranker |
                       +---------------------+
                                  |
                                  v
                  +-------------------------------+
                  | Retrieve Top-3 Documents for  |
                  | each Pipeline                 |
                  +-------------------------------+
                                  |
                                  v
                  +-------------------------------+
                  |      RRF Method for Fusion    |
                  +-------------------------------+
                                  |
                                  v
                  +-------------------------------+
                  |     Final Top-1 Document      |
                  +-------------------------------+

========================================================================================================================
"""
import os
import json
import argparse

from src.retrieve.reranker import RRF
from src.pipeline.pipelines import KelvinPipeline, JonathanPipeline, TomPipeline


if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, default="./data/datasets/preliminary/questions_example_revision.json", help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, default="./data/datasets/preliminary", help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, default="./outputs/preliminary/submission_2.json", help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--top_n', type=int, default=3, help='選擇TOPN')
    parser.add_argument('--save_each_result', type=bool, default=True, help='是否除最終 json 結果之外，同時將各個 pipeline 的 json top_n 結果也儲存')
    parser.add_argument('--yaml', type=str, default='./data/pipeline.yml', help='pipeline專用參數')  # 選擇模型
    args = parser.parse_args()  # 解析參數

    # inference with each pipeline
    answer_dict_kelvin = KelvinPipeline(args).run()
    if args.save_each_result:
        with open(args.output_path.replace(f".json", f"_kelvin_top{args.top_n}.json"), 'w', encoding='utf8') as f:
            json.dump(answer_dict_kelvin, f, ensure_ascii=False, indent=4)

    answer_dict_jonathan = JonathanPipeline(args).run()
    if args.save_each_result:
        with open(args.output_path.replace(f".json", f"_jonathan_top{args.top_n}.json"), 'w', encoding='utf8') as f:
            json.dump(answer_dict_jonathan, f, ensure_ascii=False, indent=4)

    answer_dict_tom = TomPipeline(args).run()
    if args.save_each_result:
        with open(args.output_path.replace(f".json", f"_tom_top{args.top_n}.json"), 'w', encoding='utf8') as f:
            json.dump(answer_dict_tom, f, ensure_ascii=False, indent=4)

    final_answers = {"answers": []}
    for a_kelvin, a_jonathan, a_tom in zip(answer_dict_kelvin["answers"], answer_dict_jonathan["answers"], answer_dict_tom["answers"]):
        # use RRF method to fuse each pipeline's top_n results
        top_n_lists = [a_kelvin["retrieve"], a_jonathan["retrieve"], a_tom["retrieve"]]
        rrf_fused_list = RRF(*top_n_lists, k=args.top_n)[:args.top_n]
        fused_answer = {"qid": a_kelvin['qid'], "retrieve": rrf_fused_list[0][0]}
        final_answers["answers"].append(fused_answer)

    # 將答案字典保存為json文件
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(final_answers, f, ensure_ascii=False, indent=4)
