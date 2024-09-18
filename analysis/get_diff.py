"""
Given two jsonl files of the same substask, write the output where model 1 had the correct answer and model 2 
the incorrect answer.

Also be able to provide a mcnemar table given the results
"""
import os
import sys
import json
import numpy as np
from operator import itemgetter
from typing import Dict, List
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar


EVALUATION_LOCATION = "../evaluation-pipeline-2024/results"


def parse_jsonl(path: str) -> List[Dict]:
    with open(path, "r") as f:
        ouput = json.load(f)
    return ouput


def load_model_results(*model_path_args) -> List:
    model_results = []
    for model_path in model_path_args:
        assert os.path.exists(model_path), f"Path {model_path} does not exist"
        model_results.append(parse_jsonl(model_path))
    return model_results


def get_difference(
    model_one: str,
    model_two: str,
    task: str,
    subtask: str,
):
    """Given two models and a particular task, identify the model with different predictions"""

    full_task_name = f"{task}_{subtask}.jsonl"

    model_one_path = os.path.join(EVALUATION_LOCATION, task, model_one, full_task_name)
    model_two_path = os.path.join(EVALUATION_LOCATION, task, model_two, full_task_name)

    model_one_results, model_two_results = load_model_results(
        model_one_path, model_two_path
    )
    assert len(model_one_results) == len(model_two_results)

    sys.stdout.write(
        f"Model One: {model_one}\nModel Two: {model_two}\nTask: {task}\nSubtask: {subtask}\n\n"
    )

    model_one_right = []
    model_two_right = []
    both_wrong = []
    both_right = []

    for idx in range(len(model_one_results)):
        assert model_one_results[idx]["doc_id"] == model_two_results[idx]["doc_id"]
        doc = model_one_results[idx]["doc"]
        if (
            model_one_results[idx]["acc"] == model_two_results[idx]["acc"]
            and model_one_results[idx]["acc"] == 1.0
        ):
            both_right.append(doc)
            continue
        elif (
            model_one_results[idx]["acc"] == model_two_results[idx]["acc"]
            and model_one_results[idx]["acc"] == 0.0
        ):
            both_wrong.append(doc)
        if model_one_results[idx]["acc"] == 1.0:
            model_one_right.append(doc)
        elif model_two_results[idx]["acc"] == 1.0:
            model_two_right.append(doc)
    concepts = {}
    contextTypes = {}
    contextDiffs = {}
    targetDiffs = {}

    for doc in model_one_right:
        conceptA = doc["ConceptA"]
        conceptB = doc["ConceptB"]
        contextType = doc["ContextType"]
        contextDiff = doc["ContextDiff"]
        targetDiff = doc["TargetDiff"]
        if conceptA not in concepts:
            concepts[conceptA] = [0, 0, 0, 0]
        concepts[conceptA][0] += 1
        if conceptB not in concepts:
            concepts[conceptB] = [0, 0, 0, 0]
        concepts[conceptB][0] += 1
        if contextType not in contextTypes:
            contextTypes[contextType] = [0, 0, 0, 0]
        contextTypes[contextType][0] += 1
        if contextDiff not in contextDiffs:
            contextDiffs[contextDiff] = [0, 0, 0, 0]
        contextDiffs[contextDiff][0] += 1
        if targetDiff not in targetDiffs:
            targetDiffs[targetDiff] = [0, 0, 0, 0]
        targetDiffs[targetDiff][0] += 1

    for doc in model_two_right:
        conceptA = doc["ConceptA"]
        conceptB = doc["ConceptB"]
        contextType = doc["ContextType"]
        contextDiff = doc["ContextDiff"]
        targetDiff = doc["TargetDiff"]
        if conceptA not in concepts:
            concepts[conceptA] = [0, 0, 0, 0]
        concepts[conceptA][1] += 1
        if conceptB not in concepts:
            concepts[conceptB] = [0, 0, 0, 0]
        concepts[conceptB][1] += 1
        if contextType not in contextTypes:
            contextTypes[contextType] = [0, 0, 0, 0]
        contextTypes[contextType][1] += 1
        if contextDiff not in contextDiffs:
            contextDiffs[contextDiff] = [0, 0, 0, 0]
        contextDiffs[contextDiff][1] += 1
        if targetDiff not in targetDiffs:
            targetDiffs[targetDiff] = [0, 0, 0, 0]
        targetDiffs[targetDiff][1] += 1

    for doc in both_right:
        conceptA = doc["ConceptA"]
        conceptB = doc["ConceptB"]
        contextType = doc["ContextType"]
        contextDiff = doc["ContextDiff"]
        targetDiff = doc["TargetDiff"]
        if conceptA not in concepts:
            concepts[conceptA] = [0, 0, 0, 0]
        concepts[conceptA][2] += 1
        if conceptB not in concepts:
            concepts[conceptB] = [0, 0, 0, 0]
        concepts[conceptB][2] += 1
        if contextType not in contextTypes:
            contextTypes[contextType] = [0, 0, 0, 0]
        contextTypes[contextType][2] += 1
        if contextDiff not in contextDiffs:
            contextDiffs[contextDiff] = [0, 0, 0, 0]
        contextDiffs[contextDiff][2] += 1
        if targetDiff not in targetDiffs:
            targetDiffs[targetDiff] = [0, 0, 0, 0]
        targetDiffs[targetDiff][2] += 1

    for doc in both_wrong:
        conceptA = doc["ConceptA"]
        conceptB = doc["ConceptB"]
        contextType = doc["ContextType"]
        contextDiff = doc["ContextDiff"]
        targetDiff = doc["TargetDiff"]
        if conceptA not in concepts:
            concepts[conceptA] = [0, 0, 0, 0]
        concepts[conceptA][3] += 1
        if conceptB not in concepts:
            concepts[conceptB] = [0, 0, 0, 0]
        concepts[conceptB][3] += 1
        if contextType not in contextTypes:
            contextTypes[contextType] = [0, 0, 0, 0]
        contextTypes[contextType][3] += 1
        if contextDiff not in contextDiffs:
            contextDiffs[contextDiff] = [0, 0, 0, 0]
        contextDiffs[contextDiff][3] += 1
        if targetDiff not in targetDiffs:
            targetDiffs[targetDiff] = [0, 0, 0, 0]
        targetDiffs[targetDiff][3] += 1
    """
    conceptsKeys = set(concepts.keys())
    contextTypesKeys = set(contextTypes.keys())
    contextDiffsKeys = set(contextDiffs.keys())
    targetDiffsKeys = set(targetDiffs.keys())
    for value in model_one_results:
        doc = value["doc"]
        conceptA = doc["ConceptA"]
        conceptB = doc["ConceptB"]
        contextType = doc["ContextType"]
        contextDiff = doc["ContextDiff"]
        targetDiff = doc["TargetDiff"]
        if conceptA in conceptsKeys:
            concepts[conceptA][2] += 1
        if conceptB in conceptsKeys:
            concepts[conceptB][2] += 1
        if contextType in contextTypesKeys:
            contextTypes[contextType][2] += 1
        if contextDiff in contextDiffsKeys:
            contextDiffs[contextDiff][2] += 1
        if targetDiff in targetDiffsKeys:
            targetDiffs[targetDiff][2] += 1
    """
    significant_subtasks = []
    sys.stdout.write(f"{'='*120}\n\n")
    sys.stdout.write("Concepts:\n")
    sys.stdout.write(
        f"{'Concept Name':15} | {'Model One Correct':20} | {'Model Two Correct':20} | {'Both Right':12} | {'Both Wrong':12} | {'Percentage of Total':25} | {'p-value':8}\n"
    )
    sys.stdout.write(f"{'-'*120}\n")

    for concept in concepts:
        percentage = round(
            (
                (concepts[concept][0] + concepts[concept][1])
                / (
                    concepts[concept][0]
                    + concepts[concept][1]
                    + concepts[concept][2]
                    + concepts[concept][3]
                )
            ),
            ndigits=2,
        )
        """
        2x2 array where 
        correct/correct, correct/incorrect
        incorrect/correct, incorrect/incorrect
        """
        mcnemar_table = np.array(
            [
                [concepts[concept][2], concepts[concept][0]],
                [concepts[concept][1], concepts[concept][3]],
            ]
        )

        _, p_value = mcnemar(ary=mcnemar_table, corrected=True)

        sys.stdout.write(
            f"{concept:15} | {concepts[concept][0]:20} | {concepts[concept][1]:20} | {concepts[concept][2]:12} | {concepts[concept][3]:12} | {percentage:25} | {p_value:8}\n"
        )
        if p_value < 0.05:
            significant_subtasks.append((concept, p_value))
    sys.stdout.write("\n")
    sys.stdout.write(f"{'='*120}\n\n")

    sys.stdout.write("\n")
    sys.stdout.write("Context Types:\n")
    sys.stdout.write(
        f"{'Context Name':15} | {'Model One Correct':20} | {'Model Two Correct':20} | {'Both Right':12} | {'Both Wrong':12} | {'Percentage of Total':25} | {'p-value':8}\n"
    )
    sys.stdout.write(f"{'-'*120}\n")

    for context in contextTypes:
        percentage = round(
            (
                (contextTypes[context][0] + contextTypes[context][1])
                / (
                    contextTypes[context][0]
                    + contextTypes[context][1]
                    + contextTypes[context][2]
                    + contextTypes[context][3]
                )
            ),
            ndigits=2,
        )
        mcnemar_table = np.array(
            [
                [contextTypes[context][2], contextTypes[context][0]],
                [contextTypes[context][1], contextTypes[context][3]],
            ]
        )

        _, p_value = mcnemar(ary=mcnemar_table, corrected=True)

        sys.stdout.write(
            f"{context:15} | {contextTypes[context][0]:20} | {contextTypes[context][1]:20} | {contextTypes[context][2]:12} |  {contextTypes[context][3]:12} | {percentage:25} | {p_value:8}\n"
        )
        if p_value < 0.05:
            significant_subtasks.append((context, p_value))

    sys.stdout.write("\n")
    sys.stdout.write(f"{'='*120}\n\n")

    sys.stdout.write("\n")
    sys.stdout.write("Context Diffs:\n")
    sys.stdout.write(
        f"{'Context Name':15} | {'Model One Correct':20} | {'Model Two Correct':20} | {'Both Right':12} | {'Both Wrong':12} | {'Percentage of Total':25} | {'p-value':8}\n"
    )
    sys.stdout.write(f"{'-'*120}\n")

    for context in contextDiffs:
        percentage = round(
            (
                (contextDiffs[context][0] + contextDiffs[context][1])
                / (
                    contextDiffs[context][0]
                    + contextDiffs[context][1]
                    + contextDiffs[context][2]
                    + contextDiffs[context][3]
                )
            ),
            ndigits=2,
        )
        mcnemar_table = np.array(
            [
                [contextDiffs[context][2], contextDiffs[context][0]],
                [contextDiffs[context][1], contextDiffs[context][3]],
            ]
        )

        _, p_value = mcnemar(ary=mcnemar_table, corrected=True)

        sys.stdout.write(
            f"{context:15} | {contextDiffs[context][0]:20} | {contextDiffs[context][1]:20} | {contextDiffs[context][2]:12} |  {contextDiffs[context][3]:12} | {percentage:25} | {p_value:8}\n"
        )
        if p_value < 0.05:
            significant_subtasks.append((context, p_value))

    sys.stdout.write("\n")
    sys.stdout.write(f"{'='*120}\n\n")

    sys.stdout.write("Target Diffs:\n")
    sys.stdout.write(
        f"{'Target Name':15} | {'Model One Correct':20} | {'Model Two Correct':20} | {'Both Right':12} | {'Both Wrong':12} | {'Percentage of Total':25} | {'p-value':8}\n"
    )
    sys.stdout.write(f"{'-'*120}\n")
    for target in targetDiffs:
        percentage = round(
            (
                (targetDiffs[target][0] + targetDiffs[target][1])
                / (
                    targetDiffs[target][0]
                    + targetDiffs[target][1]
                    + targetDiffs[target][2]
                    + targetDiffs[target][3]
                )
            ),
            ndigits=2,
        )

        mcnemar_table = np.array(
            [
                [targetDiffs[target][2], targetDiffs[target][0]],
                [targetDiffs[target][1], targetDiffs[target][3]],
            ]
        )

        _, p_value = mcnemar(ary=mcnemar_table, corrected=True)

        sys.stdout.write(
            f"{target:15} | {targetDiffs[target][0]:20} | {targetDiffs[target][1]:20} | {targetDiffs[target][2]:12} | {targetDiffs[target][3]:12} | {percentage:25} | {p_value:8}\n"
        )
        if p_value < 0.05:
            significant_subtasks.append((target, p_value))

    sys.stdout.write("\n")
    sys.stdout.write(f"{'='*120}\n\n")
    sys.stdout.write(f"Significant Categories:\n")
    significant_subtasks = sorted(significant_subtasks, key=itemgetter(1), reverse=True)
    for values in significant_subtasks:
        sys.stdout.write(f"{values[0]} {values[1]}\n")


def calculate_mcnemar(model_one_path, model_two_path, debug: bool = True):
    """
    Calculate the mcnemar table and the significant difference between two model's results
    """
    model_one_results, model_two_results = load_model_results(
        model_one_path, model_two_path
    )
    assert len(model_one_results) == len(model_two_results)
    y_test = np.array([1] * len(model_one_results))
    y_model_one = np.array([doc["acc"] for doc in model_one_results])
    y_model_two = np.array([doc["acc"] for doc in model_two_results])
    # for the mcnmemar table we will allow y_target to be solely the correctly predicted ones
    # (i.e. 1 for each sample) and differentiate between guesses in the models,
    # putting 0 when wrong and 1 when correct
    model_mcnemar = mcnemar_table(
        y_target=y_test, y_model1=y_model_one, y_model2=y_model_two
    )
    model_chi_squared, models_p_value = mcnemar(ary=model_mcnemar, corrected=True)
    if debug:
        sys.stdout.write(f"Model One Path: {model_one_path}\n")
        sys.stdout.write(f"Model Two Path: {model_two_path}\n")
        sys.stdout.write("\nmcnemar table:\n")
        sys.stdout.write(
            "{}\n".format(np.array2string(model_mcnemar, precision=2, separator=","))
        )
        sys.stdout.write(f"chi-squared:{model_chi_squared}\n")
        sys.stdout.write(f"p-value:{models_p_value}\n")

    model_one_acc = sum(y_model_one) / len(y_model_one)
    model_two_acc = sum(y_model_two) / len(y_model_two)
    return models_p_value, model_one_acc, model_two_acc


def compare_all_model_results(
    model_one: str, model_two: str, task_name: str = "ewok", quiet: bool = False
):
    """
    Calculate all the McNemar tables for two models and determine which tasks are significant
    """
    model_one_path = os.path.join(EVALUATION_LOCATION, task_name, model_one)
    model_two_path = os.path.join(EVALUATION_LOCATION, task_name, model_two)
    assert os.path.isdir(model_one_path), f"Path {model_one_path} does not exist"
    assert os.path.isdir(model_two_path), f"Path {model_two_path} does not exist"

    significant_differences = []

    for subtask in os.listdir(model_one_path):
        if subtask == f"{task_name}_results.json":
            continue
        if not quiet:
            sys.stdout.write(f"Task {subtask}:\n")
        model_one_task = os.path.join(model_one_path, subtask)
        model_two_task = os.path.join(model_two_path, subtask)
        p_value, model_one_acc, model_two_acc = calculate_mcnemar(
            model_one_task, model_two_task, debug=not quiet
        )
        if not quiet:
            sys.stdout.write(f"{'='*100}\n\n")

        if p_value < 0.05:
            significant_differences.append(
                (
                    subtask,
                    round(p_value, ndigits=7),
                    round(model_one_acc - model_two_acc, ndigits=2),
                )
            )
    ranked_p_values = sorted(significant_differences, key=itemgetter(1), reverse=True)
    sys.stdout.write(f"{'task_name':60}|{'p-val':10}|{'acc':4}\n")
    sys.stdout.write(f"{'='*100}\n")
    for task in ranked_p_values:
        sys.stdout.write(f"{task[0]:60}|{task[1]:10}| {task[2]:4}\n")


get_difference(
    "ELC_PaserBERT_10M_256_batch",
    "ltgbert-10m-2024",
    "ewok",
    "material-properties_filtered_results",
)
