"""
This module contains common util functions
"""
import re
from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer

import data.evaluation_results
from hir.attackers.reduction import CustomReduction


def intersection_over_union(a: Iterable[object], b: Iterable[object]) -> float:
    """
    Calculates the IOU score for two iterables

    Args:
        a: first set for IOU calculation (can be any Interable, duplicate values are ignored)
        b: second set for IOU calculation (can be any Interable, duplicate values are ignored)


    """

    set_a = set(a)
    set_b = set(b)

    # If both sets are empty return 1 (They are the same, therefore a perfect match)
    if len(set_a) == len(set_b) == 0:
        return 1

    return len(set_a & set_b) / len(set_a | set_b)


def reduction_attack(input_line: Dict[str, Any], reducer: CustomReduction) -> Tuple[int, Dict[str, Any]]:
    return input_line['id'], reducer.attack_from_json({"sentence": input_line['text']}, "tokens", "grad_input_1")


def create_analysis_dataframe(evaluation_name="all_samples_glove.json"):
    # Reading the data file via python import system
    # From the project root all directories need
    # __init__.py files to be python modules

    with path(data.evaluation_results, evaluation_name) as evaluation_path:
        df = pd.read_json(evaluation_path)

    df_clean = pd.DataFrame()

    # Adding additional fields

    # Spacy for pos and dep tags

    nlp = spacy.load("en_core_web_sm")

    # whitespace tokenizer
    nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)

    docs = list(nlp.pipe(df["full_text"]))

    import numpy as np

    # Creation of cleaned DataFrame for Analysis

    # Creation of cleaned DataFrame for Analysis
    df_clean["text"] = df['full_text']
    df_clean["l_text"] = df_clean['text'].apply(lambda x: len(x.split()))
    df_clean["relation_arguments"] = df["relation_arguments"]
    df_clean["gold_label"] = df['gold_label']
    df_clean["full_pred_label"] = df['full_pred_label']
    df_clean["full_pred_all_probs"] = df["full_pred_all_probs"]

    df_clean["hda_label"] = df['hda_label']
    df_clean["hda_pred_all_probs"] = df["hda_all_probs"]
    df_clean["mda_label"] = df['mda_label']
    df_clean["mda_pred_all_probs"] = df["mda_all_probs"]
    df_clean["dep"] = [[t.dep_ for t in doc] for doc in docs]
    df_clean["pos"] = [[t.pos_ for t in doc] for doc in docs]

    # Get model decision area (human decision area just renamed)
    df_clean["mda"] = [list(set(nt) - set(arg)) for (nt, arg) in
                       zip(df["model_nec_token_ind"], df["relation_arguments"])]
    df_clean["hda"] = df["decision_area"]

    # Length function to summarize attributes containing lists
    df_clean["l_mda"] = df_clean["mda"].apply(lambda x: len(x))
    df_clean["l_hda"] = df_clean["hda"].apply(lambda x: len(x))
    df_clean["r_l_mda"] = [row["l_mda"] / row["l_text"] for _, row in df_clean.iterrows()]
    df_clean["r_l_hda"] = [row["l_hda"] / row["l_text"] for _, row in df_clean.iterrows()]

    # Probs of decisions
    df_clean["prob_full"] = df["full_pred_prob"]
    df_clean["prob_no_mda"] = df["no_model_decision_area_prob"]
    df_clean["prob_no_hda"] = df["no_human_decision_area_prob"]
    df_clean["mda_prob"] = df["mda_prob"]
    df_clean["hda_prob"] = df["hda_prob"]

    df_clean["iou_na"] = df["iou_decision_areas_without_args"]
    df_clean["iou_full"] = df["iou_decision_areas"]

    return df_clean


labels = {}
with path(data.evaluation_results, "labels.txt") as file_path:
    with file_path.open() as file:
        for index, line in enumerate(file):
            labels[line.strip()] = index


def get_confidence(df, prob_column, prediction_label_column, ignore_sec_max=False):
    from copy import deepcopy

    df_confidence = pd.DataFrame()
    df_confidence[prob_column] = df[prob_column]
    idx_column_name = prediction_label_column + "_index"
    df_confidence[idx_column_name] = df[prediction_label_column].apply(lambda x: labels[x])

    importance = []
    for index, row in df_confidence.iterrows():
        prob_gold = row[prob_column][row[idx_column_name]]
        if ignore_sec_max:
            importance.append(prob_gold)
        else:
            max_other = deepcopy(row[prob_column])
            max_other.pop(row[idx_column_name])
            max_other = max(max_other)
            importance.append(prob_gold - max_other)

    return importance


def format_decision_area(text: str, area: List[int], placeholder: Optional[str] = None) -> str:
    tokens = text.split()
    erg = ""
    for i, token in enumerate(tokens):
        if i in area:
            erg += token if placeholder is None else len(token) * placeholder
        else:
            erg += len(token) * ("_" if placeholder is None else " ")
        erg += " "
    return erg


def print_row(cur):
    print("Text with Decision Areas:")
    print(f'org: {cur["text"]}')
    print(f'     {format_decision_area(cur["text"], cur["relation_arguments"], placeholder="^")}')
    print(f'mda: {format_decision_area(cur["text"], cur["mda"])}')
    print(f'hda: {format_decision_area(cur["text"], cur["hda"])}\n')
    print("Labels:")
    print(f'gold: {cur["gold_label"]}')
    print(f'full: {cur["full_pred_label"]}')
    print(f'only hda: {cur["hda_label"]}')
    print()
    print("Measures:")
    print(f'IOU Full: {cur["iou_full"]}')
    print(f'IOU na: {cur["iou_na"]}')
    print()
    print(f'       {[f"{num:02d}" for num in range(len(cur["full_pred_all_probs"]))]}')
    print(f'full:  {[f"{num:.2f}"[2:] for num in cur["full_pred_all_probs"]]}')
    print(f'human: {[f"{num:.2f}"[2:] for num in cur["hda_pred_all_probs"]]}')
    print(f'model: {[f"{num:.2f}"[2:] for num in cur["mda_pred_all_probs"]]}')
    print(200 * "-")
    print()
