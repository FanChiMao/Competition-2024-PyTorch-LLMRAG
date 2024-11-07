import os
import json
from src.retrieve.reranker import RRF


if __name__ == '__main__':
    # json path for the top-3 output format
    path_1 = r"D:\Others\LLM_RAG\Competition-2024-PyTorch-LLMRAG\outputs\preliminary\submission_1_jonathan_top3.json"
    path_2 = r"D:\Others\LLM_RAG\Competition-2024-PyTorch-LLMRAG\outputs\preliminary\submission_1_tom_top3.json"
    path_3 = r"D:\Others\LLM_RAG\Competition-2024-PyTorch-LLMRAG\outputs\preliminary\submission_1_kelvin_top3.json"
    path_4 = r"D:\Others\LLM_RAG\Competition-2024-PyTorch-LLMRAG\outputs\preliminary\submission_1_edward_top3.json"
    output_path = "D:\Others\LLM_RAG\Competition-2024-PyTorch-LLMRAG\outputs\preliminary\submission_1_K60.json"

    with open(path_1, 'rb') as f:
        path_1_answer = json.load(f)
    with open(path_2, 'rb') as f:
        path_2_answer = json.load(f)
    with open(path_3, 'rb') as f:
        path_3_answer = json.load(f)
    with open(path_3, 'rb') as f:
        path_4_answer = json.load(f)

    final_answers = {"answers": []}
    for qid, (a_kelvin, a_jonathan, a_tom, a_edward) in enumerate(zip(path_1_answer["answers"], path_2_answer["answers"], path_3_answer["answers"], path_4_answer["answers"])):
        # use RRF method to fuse each pipeline's top_n results
        top_n_lists = [a_kelvin["retrieve"], a_jonathan["retrieve"], a_tom["retrieve"]]
        print("QID: ", qid+1)
        rrf_fused_list = RRF(*top_n_lists, k=60, print_score=True)[:3]
        fused_answer = {"qid": a_kelvin['qid'], "retrieve": rrf_fused_list[0][0]}
        final_answers["answers"].append(fused_answer)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(final_answers, f, ensure_ascii=False, indent=4)