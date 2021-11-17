import heapq
import time
from copy import deepcopy
from threading import Lock
from typing import List, Tuple

import numpy as np
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.fields import SequenceLabelField, SpanField, TextField
from allennlp.interpret.attackers import utils
from allennlp.interpret.attackers.attacker import Attacker
from allennlp.predictors import Predictor

DELETED_TOKENS = "deleted_toktens"


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print(f'{method.__name__}  {(te - ts) * 1000} ms')
        return result

    return timed


def instance_has_changed(instance: Instance, fields_to_compare: JsonDict):
    field = "labels"
    if instance[field] != fields_to_compare[field]:
        return True
    return False


lock = Lock()


@Attacker.register("custom-reduction")
class CustomReduction(Attacker):
    """
    Runs the input reduction method from [Pathologies of Neural Models Make Interpretations
    Difficult](https://arxiv.org/abs/1804.07781), which removes as many words as possible from
    the input without changing the model's prediction.
    The functions on this class handle a special case for NER by looking for a field called "tags"
    This check is brittle, i.e., the code could break if the name of this field has changed, or if
    a non-NER model has a field called "tags".
    Registered as an `Attacker` with name "input-reduction".
    """

    def __init__(self, predictor: Predictor, beam_size: int = 2) -> None:
        super().__init__(predictor)
        self.beam_size = beam_size

    def attack_from_json(
            self,
            inputs: JsonDict,
            input_field_to_attack: str = "tokens",
            grad_input_field: str = "grad_input_1",
            ignore_tokens: List[str] = None,
            target: JsonDict = None,
    ):
        if target is not None:
            raise ValueError("Input reduction does not implement targeted attacks")
        ignore_tokens = ["@@NULL@@"] if ignore_tokens is None else ignore_tokens
        with lock:
            original_instances = self.predictor.json_to_labeled_instances(inputs)
        original_text_field: TextField = original_instances[0][input_field_to_attack]  # type: ignore
        original_tokens = deepcopy(original_text_field.tokens)
        final_tokens_deleted = []
        for instance in original_instances:
            final_tokens_deleted.append(
                self._attack_instance(inputs, instance, input_field_to_attack, grad_input_field, ignore_tokens)
            )

        for i, (tokens, deleted) in enumerate(final_tokens_deleted):
            ind = list(range(-1, len(original_tokens) - 1))
            normalized_deleted_indices = []
            for d in deleted:
                normalized_deleted_indices += [ind.pop(d)]
            final_tokens_deleted[i] = (tokens, normalized_deleted_indices)

        argument_spans = []
        argument_spans += list(range(instance["e1"].span_start - 1, instance["e1"].span_end))
        argument_spans += list(range(instance["e2"].span_start - 1, instance["e2"].span_end))

        return sanitize(
            {
                "final": [t[0] for t in final_tokens_deleted],
                "original": original_tokens,
                "deleted": [t[1] for t in final_tokens_deleted],
                "nec_tokens": ind[1:],
                "model_decision_area": set(ind[1:]) - set(argument_spans),
                "relation_arguments": argument_spans
            }
        )

    def _attack_instance(
            self,
            inputs: JsonDict,
            instance: Instance,
            input_field_to_attack: str,
            grad_input_field: str,
            ignore_tokens: List[str],
    ):
        # Save fields that must be checked for equality
        fields_to_compare = utils.get_fields_to_compare(inputs, instance, input_field_to_attack)

        # Set num_ignore_tokens, which tells input reduction when to stop
        # We keep at least one token for input reduction on classification/entailment/etc.

        num_ignore_tokens = 3

        text_field: TextField = instance[input_field_to_attack]  # type: ignore
        current_tokens = deepcopy(text_field.tokens)
        current_label = instance["labels"].label
        deleted = []
        candidates = [(instance, -1)]
        # keep removing tokens until prediction is about to change
        while len(current_tokens) > num_ignore_tokens and candidates:
            def get_length(input_instance: Instance):
                input_text_field: TextField = input_instance[input_field_to_attack]  # type: ignore
                return len(input_text_field.tokens)

            # sort current candidates by smallest length (we want to remove as many tokens as possible)
            candidates = heapq.nsmallest(self.beam_size, candidates, key=lambda x: get_length(x[0]))

            beam_candidates = deepcopy(candidates)
            candidates = []
            for beam_instance, smallest_idx in beam_candidates:
                # get gradients and predictions

                with lock:
                    grads, outputs = self.predictor.get_gradients([beam_instance])

                for output in outputs:
                    if isinstance(outputs[output], torch.Tensor):
                        outputs[output] = outputs[output].detach().cpu().numpy().squeeze().squeeze()
                    elif isinstance(outputs[output], list):
                        outputs[output] = outputs[output][0]

                # relabel beam_instance since last iteration removed an input token
                with lock:
                    beam_instance = self.predictor.predictions_to_labeled_instances(beam_instance, outputs)[0]
                if instance_has_changed(beam_instance, fields_to_compare):
                    continue

                # remove a token from the input
                text_field: TextField = beam_instance[input_field_to_attack]  # type: ignore
                current_tokens = deepcopy(text_field.tokens)
                if DELETED_TOKENS in beam_instance["metadata"].metadata:
                    deleted = deepcopy(beam_instance["metadata"].metadata[DELETED_TOKENS])
                reduced_instances_and_smallest = _remove_one_token(
                    beam_instance,
                    input_field_to_attack,
                    grads[grad_input_field][0],
                    self.beam_size,  # type: ignore
                )
                candidates.extend(reduced_instances_and_smallest)
        return current_tokens, deleted


def _remove_one_token(
        instance: Instance,
        input_field_to_attack: str,
        grads: np.ndarray,
        beam_size: int,
) -> List[Tuple[Instance, int, List[int]]]:
    """
    Finds the token with the smallest gradient and removes it.
    """
    # Compute L2 norm of all grads.
    grads_mag = [np.sqrt(grad.dot(grad)) for grad in grads] # TODO np.linalg.norm ?

    # TODO ändert sich das in jeder iteration? Weil ich tokens entferne?
    get_ent_span = lambda x: (x.span_start, x.span_end)
    e1_span = get_ent_span(instance["e1"])
    e2_span = get_ent_span(instance["e2"])

    # Keep the [CLS] token and the relation arguments
    # Skip all ignore_tokens by setting grad to infinity
    mandatory_args = (
            [0] + list(range(e1_span[0], e1_span[1] + 1)) + list(range(e2_span[0], e2_span[1] + 1))
    )
    for relation_argument_token_index in mandatory_args:
        grads_mag[relation_argument_token_index] = float("inf")

    reduced_instances_and_smallest: List[Tuple[Instance, int, List[int]]] = []
    for _ in range(beam_size):
        # copy instance and edit later
        copied_instance = deepcopy(instance)
        copied_text_field: TextField = copied_instance[input_field_to_attack]  # type: ignore

        # find smallest
        smallest = np.argmin(grads_mag) # TODO drei kleinsten holen und in die liste packen...
        if grads_mag[smallest] == float("inf"):  # if all are ignored tokens, return.
            break
        grads_mag[smallest] = float("inf")  # so the other beams don't use this token

        # remove smallest
        inputs_before_smallest = copied_text_field.tokens[0:smallest]
        inputs_after_smallest = copied_text_field.tokens[smallest + 1:]
        copied_text_field.tokens = inputs_before_smallest + inputs_after_smallest

        get_span_field = lambda x: SpanField(x[0] - 1, x[1] - 1, copied_instance.fields["tokens"])
        if smallest < e1_span[0]:
            copied_instance.add_field("e1", get_span_field(e1_span))

        if smallest < e2_span[0]:
            copied_instance.add_field("e2", get_span_field(e2_span))

        if DELETED_TOKENS not in copied_instance["metadata"].metadata:
            copied_instance["metadata"].metadata[DELETED_TOKENS] = [smallest]
        else:
            copied_instance["metadata"].metadata[DELETED_TOKENS] += [smallest]

        copied_instance.indexed = False
        reduced_instances_and_smallest.append((copied_instance, smallest))

    return reduced_instances_and_smallest


def _get_ner_tags_and_mask(instance: Instance, input_field_to_attack: str, ignore_tokens: List[str]):
    """
    Used for the NER task. Sets the num_ignore tokens, saves the original predicted tag and a 0/1
    mask in the position of the tags
    """
    # Set num_ignore_tokens
    num_ignore_tokens = 0
    input_field: TextField = instance[input_field_to_attack]  # type: ignore
    for token in input_field.tokens:
        if str(token) in ignore_tokens:
            num_ignore_tokens += 1

    # save the original tags and a 0/1 mask where the tags are
    tag_mask = []
    original_tags = []
    tag_field: SequenceLabelField = instance["tags"]  # type: ignore
    for label in tag_field.labels:
        if label != "O":
            tag_mask.append(1)
            original_tags.append(label)
            num_ignore_tokens += 1
        else:
            tag_mask.append(0)
    return num_ignore_tokens, tag_mask, original_tags
