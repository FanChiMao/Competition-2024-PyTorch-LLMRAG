import json
import argparse
import os

from src.pipeline.pipelines import BasePipeline, KelvinPipeline, JonathanPipeline, EdwardPipeline, TomPipeline


if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, default="./data/datasets/preliminary/questions_example_revision.json", help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, default="./data/datasets/preliminary", help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, default="./outputs/submission.json", help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--top_n', type=int, default=1, help='選擇TOPN')
    parser.add_argument('--name', type=str, default='edward', help='選擇方法')
    parser.add_argument('--yaml', type=str, default='./data/pipeline.yml', help='pipeline專用參數')  # 選擇模型
    args = parser.parse_args()  # 解析參數

    # choose method
    if args.name == 'baseline':
        answer_dict = BasePipeline(args).run()
    elif args.name == 'kelvin':
        answer_dict = KelvinPipeline(args).run()
    elif args.name == 'jonathan':
        answer_dict = JonathanPipeline(args).run()
    elif args.name == 'edward':
        answer_dict = EdwardPipeline(args).run()
    elif args.name == 'tom':
        answer_dict = TomPipeline(args).run()
    else:
        raise ValueError(f"Invalid name: {args.name}, only support 'baseline', 'kelvin', 'jonathan', 'edward', 'tom'!")

    # pick top 1
    for item in answer_dict["answers"]:
        item["retrieve"] = item["retrieve"][0]

    # 將答案字典保存為json文件
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符

    print(f"The result json file has been saved in {os.path.abspath(args.output_path)}")
