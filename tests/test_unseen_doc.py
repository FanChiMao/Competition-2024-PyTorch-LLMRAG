import json


if __name__ == '__main__':
    json_file = "../data/datasets/preliminary/questions_example_revision.json"

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            qs_ref = json.load(f)  # Read the questions file

        doc_id = []
        for data in qs_ref.get('questions', []):
            source = data['source']
            if data['category'] == 'finance':
                if isinstance(source, list):  # Check if source is a list
                    doc_id.extend(source)  # Add elements from the list to doc_id
                elif isinstance(source, int):  # If source is a single integer
                    doc_id.append(source)

        # Create a set of unique integers from doc_id
        doc_id_set = set(doc_id)

        # Find missing integers from 0 to 1034
        full_set = set(range(1035))
        unseen_id = sorted(full_set - doc_id_set)
        print("Appear doc Id in finance:", doc_id)
        print("Unseen doc Id in finance:", set(unseen_id))
        print(len(doc_id))
        print(len(unseen_id))
        print(len(full_set))
    except FileNotFoundError:
        print(f"The file '{json_file}' was not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON file. Please check the file format.")

