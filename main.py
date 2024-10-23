import json
import argparse
from src.pipeline.base import BasePipeline
from src.pipeline.pipelines import KelvinPipeline

if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, default="./data/dataset/preliminary/questions_example_revision.json", help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, default="./data/reference", help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, default="./output/submission.json", help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--top_n', type=int, default=1, help='選擇TOPN')
    parser.add_argument('--yaml', type=str, default='./data/pipeline.yml', help='pipeline專用參數')  # 選擇模型
    args = parser.parse_args()  # 解析參數
    
    answer_dict = BasePipeline(args).run()      # 執行BasePipeline
    answer_dict = KelvinPipeline(args).run()    # 執行KelvinPipeline

    # TODO: ensemble
    
    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符