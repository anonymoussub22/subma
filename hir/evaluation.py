import collections
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Set, Union

import click
import numpy as np
import pandas as pd
import srsly
import torch
from allennlp.common.util import get_spacy_model
from allennlp.models import load_archive
from tqdm import tqdm

import pretrained_models
from data import decision_area
from hir.attackers.reduction import CustomReduction
from hir.predictor import ClassifierPredictor
from hir.util import intersection_over_union, reduction_attack

# Default file names
TEST_KEY_FILE_NAME = "CORRECTED_TEST_FILE_KEY.TXT"
SCORER_FILE_NAME = "semeval2010_task8_scorer-v1.2.pl"
MODEL_FILE_NAME = "model_glove.tar.gz"
TEST_SAMPLES_FILES = "TEST_FILE_CLEAN.TXT"
PARSED_FILE = "test.txt.dep"
TEST_DATA_FILE = "test.txt"
DECISION_AREA_RAP = "all.jsonl"


GOLD_LABEL = "gold_label"
ID = "id"
TEXT = "text"
DEC_AREA = "dec_area"


class EvaluationWithoutTestSamplesError(Exception):
    pass


class NoDecisionAreaAnnotatedError(Exception):
    pass


class NoArgumentAnnotationError(Exception):
    pass


def get_current_time_formatted():
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def normalize_text(text: str) -> str:
    """
    Normalize a RC input to ensure Whitespace-Tokenization works as intended.
    This especially means argument indicators (<e1>, ...) are surrounded by Whitespaces.
    We remove multi-whitespaces by single whitespaces!

    :param test:
    :return:
    """

    text = re.sub("((?:</?e\d>)|[\.\,:;'])", " \g<1> ", text)
    text = re.sub(" +", " ", text)

    if text.startswith('"'):
        text = text[1:]

    if text.endswith('"'):
        text = text[:-1]

    text = text.strip()

    return text


class Evaluator:
    def predict_label_for_sample(self, id, text, reduction: Union[Set, str] = None):
        prediction = self.predictor.predict_reduction(text, red_indices=reduction)

        logits = np.array(prediction["logits"])
        pred_id = np.argmax(logits)

        return (
            self.archive.model.vocab.get_token_from_index(pred_id, namespace="labels"),
            prediction["probabilities"][pred_id],
            prediction["probabilities"]
        )


def specified_or_default_path(path_candidate: Union[str, Path], data_module, file_name: str):
    if path_candidate is None:
        with path(data_module, file_name) as default_path:
            return default_path.absolute()

    return path_candidate


def convert_decision_area_annotation(annotation: dict, include_all: bool = False) -> Dict[str, Any]:
    """
    Given a RAP annotation in JSONL format, produces sample in semeval 2020 task 8 format.
    The token indices in 2nd result relate to the token indices without enitity tags (<e1> ...)

    :param annotation:
    :return:
    """
    decision_area_annotation = [l for l in annotation["label"] if l[2] == "DEC_AREA"]
    no_annotation_annotation = [l for l in annotation["label"] if l[2] == "NO_ANNOTATIONS"]
    no_arg_text = annotation["data"].split("\n")[0]

    if not (include_all or decision_area_annotation or no_annotation_annotation):
        raise NoDecisionAreaAnnotatedError(f"Sample {annotation} has no annotated decision area.")

    decision_areas = []
    for l in decision_area_annotation:
        text_until_decision_area_index = l[0]
        decision_area = no_arg_text[l[0] : l[1]]

        if decision_area.startswith(" "):
            text_until_decision_area_index += 1

        token_index = no_arg_text[:text_until_decision_area_index].count(" ")
        decision_areas.append(tuple(range(token_index, token_index + decision_area.strip().count(" ") + 1)))
        #decision_areas.append((token_index, token_index + decision_area.strip().count(" ")))

    try:
        e1_annotation = [l for l in annotation["label"] if l[2] == "E1"][0]
        e2_annotation = [l for l in annotation["label"] if l[2] == "E2"][0]
        text = (
            f"{no_arg_text[:e1_annotation[0]]} <e1> {no_arg_text[e1_annotation[0]:e1_annotation[1]]} </e1> "
            f"{no_arg_text[e1_annotation[1]: e2_annotation[0]]} <e2> "
            f"{no_arg_text[e2_annotation[0]: e2_annotation[1]]} </e2> {no_arg_text[e2_annotation[1]:]}"
        )
        text = re.sub(" +", " ", text)
    except:
        raise NoArgumentAnnotationError()

    label_regex = r"---+\n(.+)\n"
    match = re.search(label_regex, annotation["data"])
    label = ""
    if match:
        label = match.group(1)

    return {ID: annotation["id"], TEXT: text, DEC_AREA: decision_areas, GOLD_LABEL: label}


def read_decision_area_file(decision_area_path: Union[Path, str], include_all: bool = False) -> List[Dict[str, Any]]:
    input_lines = []
    das = list(srsly.read_jsonl(decision_area_path))
    for sample in das:
        try:
            input_lines += [convert_decision_area_annotation(sample, include_all=include_all)]

        except NoDecisionAreaAnnotatedError:
            continue
        except NoArgumentAnnotationError:
            continue

    return input_lines


class DecisionAreaEvaluator(Evaluator):
    def __init__(self, decision_area_path: Union[str, Path] = None, include_all: bool = False):
        self.decision_area_path = specified_or_default_path(decision_area_path, decision_area, DECISION_AREA_RAP)
        self.input_lines: List[Dict[str, Any]] = read_decision_area_file(self.decision_area_path, include_all=include_all)
        self.nlp = get_spacy_model("en_core_web_trf")
        self.model_path = None
        self.archive = None
        self.predictor = None

    def load_model(self, model_path: Union[Path, str] = None, cuda_device = None):
        self.model_path = specified_or_default_path(model_path, pretrained_models, MODEL_FILE_NAME)
        if cuda_device is None:
            self.archive = load_archive(
                self.model_path, cuda_device=torch.cuda.current_device() if torch.cuda.is_available() else None
            )
        else:
            self.archive = load_archive(
                self.model_path, cuda_device=cuda_device
            )
        self.predictor = ClassifierPredictor(self.archive.model, self.archive.dataset_reader)

    def evaluate_sample(self, text, decision_area, eval_complete=True):

        pred_label, pred_prob, all_probs = None, None, None

        if eval_complete:
            pred_label, pred_prob, all_probs = self.predict_label_for_sample(-1, text)

        red_pred_label, red_pred_prob, red_all_probs = self.predict_label_for_sample(-1, text, reduction=decision_area)

        return (pred_label, pred_prob, all_probs), (red_pred_label, red_pred_prob, red_all_probs)

    def perform_attacking(self, file_prefix="attacker_output", beam_size=2):
        reduced_by_id: List[Dict[Any, Any]] = []
        reducers = len(self.input_lines) * [CustomReduction(self.predictor, beam_size=beam_size)]
        length = len(self.input_lines)
        with ThreadPoolExecutor(max_workers=1) as executor:
            for idx, cur_reduced in tqdm(executor.map(reduction_attack, self.input_lines, reducers), total=length):
                cur_reduced[ID] = idx  # Add rap ID
                reduced_by_id.append(cur_reduced)

        srsly.write_jsonl(f"{file_prefix}_{get_current_time_formatted()}.jsonl", reduced_by_id)

    @staticmethod
    def read_attacks_from_file(path: str) -> Dict[int, Dict[str, Any]]:
        reduced: Dict[int, Dict[str, Any]] = dict()
        for line in srsly.read_jsonl(path):
            reduced[line[ID]] = line

        return reduced

    def evaluate_all_samples(self, reduced_dict):

        df_data = collections.defaultdict(list)

        # analyse mit spacy

        for input_line in tqdm(self.input_lines):
            rap_idx = input_line[ID]
            try:
                reduced = reduced_dict[rap_idx]
            except KeyError as e:
                print(f"No reduced output found for RAP ID '{rap_idx}': continue", file=sys.stderr)
                continue

            text = input_line[TEXT]
            decision_area = input_line[DEC_AREA]

            unique_decision_area = list({i for r in decision_area for i in r})

            (pred_label, pred_prob, all_probs), (no_human_decision_area, no_human_decision_area_prob, no_human_decision_area_all_probs) = self.evaluate_sample(
                text, unique_decision_area
            )

            _, (no_model_decision_area, no_model_decision_area_prob, no_model_decision_area_all_probs) = self.evaluate_sample(text, reduced["model_decision_area"], eval_complete=False)

            _, (only_model_decision_area_label, only_model_decision_area_prob, only_model_decision_area_all_probs) = self.evaluate_sample(text, reduced["deleted"][0], eval_complete=False)
            df_data["mda_label"] += [only_model_decision_area_label]
            df_data["mda_prob"] += [only_model_decision_area_prob]
            df_data["mda_all_probs"] += [only_model_decision_area_all_probs]

            all_indices = set(range(len(reduced["original"]) - 1))

            only_human_decision_area_reduction = list(all_indices - set(unique_decision_area) - set(reduced["relation_arguments"]))
            _, (only_hda_label, only_hda_prob, only_hda_all_probs) = self.evaluate_sample(text, only_human_decision_area_reduction, eval_complete=False)
            df_data["hda_label"] += [only_hda_label]
            df_data["hda_prob"] += [only_hda_prob]
            df_data["hda_all_probs"] += [only_hda_all_probs]

            max_reduction = list(all_indices - set(reduced["relation_arguments"]))
            _, (max_red_label, max_red_prob, max_red_all_probs) = self.evaluate_sample(text, max_reduction, eval_complete=False)
            df_data["max_red_label"] += [max_red_label]
            df_data["max_red_prob"] += [max_red_prob]
            df_data["max_red_all_probs"] += [max_red_all_probs]

            # red_random reduction areas:
            random_decision_areas = []

            rda_label = []
            rda_prob = []
            rda_all_probs = []

            no_rda_label = []
            no_rda_prob = []
            no_rda_all_probs = []

            NUM_RAND_RED_PER_SAMPLE = 3
            candidates = all_indices - set(reduced["relation_arguments"])
            for i in range(NUM_RAND_RED_PER_SAMPLE):
                k = min(len(reduced["model_decision_area"]), len(candidates) -1)
                rand_da = set(random.sample(candidates, k))
                rand_red = list(candidates - rand_da)
                _, (only_rda_label, only_rda_prob, only_rda_all_probs) = self.evaluate_sample(text, rand_red, eval_complete=False)
                random_decision_areas += [rand_da]
                rda_label += [only_rda_label]
                rda_prob += [only_rda_prob]
                rda_all_probs += [only_rda_all_probs]

                _, (rem_rda_label, rem_rda_prob, rem_rda_all_probs) = self.evaluate_sample(text, rand_da, eval_complete=False)
                no_rda_label += [rem_rda_label]
                no_rda_prob += [rem_rda_prob]
                no_rda_all_probs += [rem_rda_all_probs]

            df_data["random_decision_areas"] += [list(random_decision_areas)]
            df_data["rda_label"] += [list(rda_label)]
            df_data["rda_prob"] += [list(rda_prob)]
            df_data["rda_all_probs"] += [list(rda_all_probs)]

            df_data["no_rda_label"] += [list(no_rda_label)]
            df_data["no_rda_prob"] += [list(no_rda_prob)]
            df_data["no_rda_all_probs"] += [list(no_rda_all_probs)]

            # These evaluations on random decision areas !
            df_data["full_text"] += [" ".join(reduced["original"][1:])]
            df_data["full_arg_annotated_text"] += [text]
            df_data["gold_label"] += [input_line[GOLD_LABEL]]
            df_data["full_pred_label"] += [pred_label]
            df_data["full_pred_prob"] += [pred_prob]
            df_data["full_pred_all_probs"] += [all_probs]

            # Reduction without decision area, min model decision area
            df_data["reduced_text"] += [" ".join(reduced["final"][0][1:])]
            df_data["model_nec_token_ind"] += [sorted(reduced["nec_tokens"])]
            df_data["no_model_decision_area_label"] += [no_model_decision_area]
            df_data["no_model_decision_area_prob"] += [no_model_decision_area_prob]
            df_data["no_model_decision_area_all_prob"] += [no_model_decision_area_all_probs]

            # Token ID of relation arguments
            df_data["relation_arguments"] += [sorted(reduced["relation_arguments"])]

            # Token IDs der decision area
            df_data["decision_area"] += [sorted(unique_decision_area)]
            df_data["no_human_decision_area_label"] += [no_human_decision_area]
            df_data["no_human_decision_area_prob"] += [no_human_decision_area_prob]
            df_data["no_human_decision_area_all_probs"] += [no_human_decision_area_all_probs]

            # Human equvalent to model_nec_token_ind
            df_data["human_nec_token_ind"] += [sorted(reduced["relation_arguments"] + unique_decision_area)]

            # Statistiken über Decision Areas (am besten sind unsere größer als die des Models?)
            # Anteil Decision Area vom Satz
            # Welcher Teil vom Satz ist decision Area (POS und DEP Tags)

            # Relative share of the human Decision Area in the text
            df_data["relative_human_nec_token_len"] += [
                len(df_data["human_nec_token_ind"][-1]) / len(reduced["original"][1:])
            ]
            # Relative share of the calculated Decision Area in the text
            df_data["relative_no_human_nec_token_len"] += [
                len(df_data["model_nec_token_ind"][-1]) / len(reduced["original"][1:])
            ]

            # Überdeckung model und human decision area
            # Es ist O.K. wenn model decision area eine echte Teilmenge der human decision area ist
            df_data["iou_decision_areas"] += [
                intersection_over_union(df_data["model_nec_token_ind"][-1], df_data["human_nec_token_ind"][-1])
            ]
            df_data["iou_decision_areas_without_args"] += [
                intersection_over_union(
                    set(df_data["model_nec_token_ind"][-1]) - set(reduced["relation_arguments"]),
                    set(unique_decision_area) - set(reduced["relation_arguments"]),
                )
            ]
            df_data["is_model_decision_area_subset"] += [
                set(df_data["model_nec_token_ind"][-1]).issubset(df_data["human_nec_token_ind"][-1])
            ]
            df_data["is_human_decision_area_subset"] += [
                set(df_data["human_nec_token_ind"][-1]).issubset(df_data["model_nec_token_ind"][-1])
            ]

            # Analyse Idee: POS und Dependency Tags der Decision Areas
            # Confusion Matrizen analysieren
            # Performance ohne Decision Area (F1 Score per label)

            # Einfluss der Decision Area unterschiedlich für korrekt und falsch klassifizierte Beispiele
            # Decision Area für verschiedene gold Labels

            # Wir brauchen ein Muster in den Decision Area-Differenzen, gibt es das?
            # Ist das Muster für verschiedene DL Ansätze unterschiedlich

            # Hypothesen:
            # 1. Decision Areas geben Aufschluss über falsch gelernte Pattern
            # 2. Durch Decision Area finden wir für das Model problematische Formulierungen
            # 2.1. Bestimmte Formulierungen werden unrobust vom Model verarbeitet
            # 2.1. -> Schwächen genau bekannt und Optimierungspotential wird schnell Sichtbar

        df = pd.DataFrame.from_dict(df_data)

        df.to_json("all_samples.json")

    @property
    def is_initialized(self):
        return self.archive is not None


@click.command()
@click.option("--attack", "-a", is_flag=True, default=False, help="Flag to set program in attacking mode.")
@click.option(
    "--attack_file_prefix",
    "-afp",
    type=str,
    help="File prefix for the attack results, which is stored in the working directory.",
)
@click.option(
    "--eval",
    "-e",
    type=str,
    help="Implicit setting program to evaluation mode. Requires the path to the attacker file!",
)
@click.option(
    "--test_file", type=str, default=None, help="Sets the test file. Should be the same for attacker and evaluation."
)
@click.option(
    "--model", type=str, default=None, help="Sets the model. Should be the same for attacker and evaluation."
)
@click.option(
    "--beam_size",
    "-bs",
    type=int,
    default=2,
    help="Beam-size for attack reducer. High values -> more specific results and higher runtime",
)
@click.option(
    "--cuda_device",
    "-cd",
    type=int,
    default=None,
    help="Optinal Cuda device",
)
def main(attack: bool, attack_file_prefix: str, eval: str, test_file: str, model: str, beam_size: int, cuda_device: int):
    if not attack and not eval:
        raise Exception("Please provide command for 'attack' or 'eval'!")

    print("Script started")
    if attack:
        dae = DecisionAreaEvaluator(test_file, include_all=True)
    if eval:
        dae = DecisionAreaEvaluator(test_file, include_all=False)
    print("Load Model")
    dae.load_model(model, cuda_device=cuda_device)

    if attack:
        print("Start attacking")
        if attack_file_prefix:
            dae.perform_attacking(attack_file_prefix, beam_size=beam_size)
        else:
            dae.perform_attacking(beam_size=beam_size)
    if eval:
        reduced = DecisionAreaEvaluator.read_attacks_from_file(eval)
        dae.evaluate_all_samples(reduced)


if __name__ == "__main__":
    main()
