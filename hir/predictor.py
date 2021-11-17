from typing import Dict, List

import numpy
import numpy as np
from allennlp.common import JsonDict
from allennlp.common.util import import_module_and_submodules
from allennlp.data.fields import LabelField
from allennlp.interpret.saliency_interpreters import SimpleGradient
from allennlp.models import load_archive
from allennlp.predictors import Predictor
from overrides import overrides
from traitlets import Instance

from hir.attackers.reduction import CustomReduction


@Predictor.register("classifier", exist_ok=True)
class ClassifierPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parsed_texts = {}

    def predict(self, sentence: str) -> JsonDict:
        # This method is implemented in the base class.
        return self.predict_json({"sentence": sentence})

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:

        logits = outputs["logits"]

        if isinstance(outputs["logits"], list):
            logits = np.array(logits)

        label_index = logits.argmax()
        label = self._model.vocab.get_token_from_index(label_index, namespace="labels")



        instance.add_field("labels", LabelField(label, label_namespace="labels"), self._model.vocab)
        # instance.index_fields(self._model.vocab)
        return [instance]

    def predict_reduction(self, text, red_indices=None):
        reduction = red_indices
        if red_indices is not None:
            if isinstance(red_indices, list):
                reduction = red_indices

        instance = self._dataset_reader.text_to_instance(text=text, rediction=reduction)

        return self.predict_instance(instance)

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:

        text = json_dict["sentence"]

        return self._dataset_reader.text_to_instance(text=text)
