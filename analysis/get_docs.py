"""
Given particular key value pairs for a task and subtask name, 
return all the documents that match the description
"""
import os
import json
from typing import List, Dict

EVALUATION_LOCATION = "../evaluation-pipeline-2024/results"


def parse_jsonl(path: str) -> List[Dict]:
    with open(path, "r") as f:
        ouput = json.load(f)
    return ouput


def get_docs(
    model_name: str,
    subtask: str,
    filters: Dict,
    task: str = "ewok",
    eval_loc: str = EVALUATION_LOCATION,
):
    filename = f"{task}_{subtask}.jsonl"
    doc_path = os.path.join(eval_loc, task, model_name, filename)
    all_docs = parse_jsonl(doc_path)

    filtered_docs = []

    for doc_value in all_docs:
        doc = doc_value["doc"]
        add = True
        for filter in filters:
            if filter not in doc:
                add = False
                break
            elif filter in doc and doc[filter] != filters[filter]:
                add = False
                break
        if add:
            filtered_docs.append(doc)
    return filtered_docs


if __name__ == "__main__":
    docs = get_docs(
        "ELC_PaserBERT_10M_256_batch",
        "material-properties_filtered_results",
        {"ConceptB": "warm"},
    )

    for doc in docs:
        print(doc)
