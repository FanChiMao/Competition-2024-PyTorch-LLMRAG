import json
from pathlib import Path

# FINANCE_CORPUS_WITH_OCR_PATH = Path("finance_corpus_dict_with_ocr.json")
FINANCE_CORPUS_WITHOUT_OCR_PATH = Path("finance_corpus_dict_no_ocr.json")
GROUND_TRUTHS_PATH = Path("../dataset/preliminary/ground_truths_example_revision.json")
QUESTION_SOURCE_PATH = Path("../dataset/preliminary/questions_example_revision.json")

def main() -> None:
    with open(FINANCE_CORPUS_WITHOUT_OCR_PATH, 'rb+') as fs:
        corpus_dict_finance = {int(k): v for k, v in json.load(fs).items() if v == ""}

    with open(QUESTION_SOURCE_PATH, 'rb+') as fs:
        questions = json.load(fs)
        questions = questions["questions"]

    with open(GROUND_TRUTHS_PATH, 'rb+') as fs:
        ground_truths = json.load(fs)
        ground_truths = ground_truths["ground_truths"]

    table_only_finance_file = list(corpus_dict_finance.keys())
    use_as_source_count : int = 0
    use_as_answer_count : int = 0

    for question, ground_truth in zip(questions, ground_truths):
        if question["category"] == "finance":
            is_source_in_table : bool = False
            is_ground_truth_in_table : bool = (ground_truth["retrieve"] in table_only_finance_file)

            for source in question["source"]:
                if source in table_only_finance_file:
                    use_as_source_count += 1
                    is_source_in_table = True
                    break

            qid: int = question["qid"]
            if is_ground_truth_in_table and is_source_in_table:
                use_as_answer_count += 1
                print(f"qid: {qid} use table file as source and answer!!!")
            elif is_source_in_table:
                print(f"qid: {qid} use table file as source!!!")

    print(f"{use_as_answer_count} / {use_as_source_count}")

if __name__ == '__main__':
    main()
