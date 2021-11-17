from typing import Dict

import torch
import torch.nn as nn
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.fields import MetadataField
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure


def get_mask_for_spans(spans, shape):
    mask = torch.zeros(shape, device=spans.device)
    for batch in range(shape[0]):
        mask[batch][spans[batch][0] : spans[batch][1] + 1] = 1

    return mask


@Model.register("graph_conv")
class ReAgcn(Model):
    """
    Simple GCN layer
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        pooler: Seq2VecEncoder,
        encoder: Seq2SeqEncoder = None,
        hidden_size: int = 1024,
        hidden_dropout_prob: float = 0.1,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.embedder = embedder
        self.pooler = pooler
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = nn.Linear(hidden_size * 3, num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.label_tokens = vocab.get_index_to_token_vocabulary("labels")
        self.loss = torch.nn.CrossEntropyLoss()
        self.acc = CategoricalAccuracy()
        evaluation_labels = [
            k for (k, v) in self.vocab.get_index_to_token_vocabulary(namespace="labels").items() if v != "Other"
        ]
        self.f1 = FBetaMeasure(labels=evaluation_labels, average="micro")

        initializer(self)

    def forward(  # type: ignore
        self,
        tokens: TextFieldTensors,
        e1,
        e2,
        labels: torch.IntTensor = None,
        metadata: MetadataField = None,
    ) -> Dict[str, torch.Tensor]:

        # min_e1 = torch.min(e1, dim=1).values
        # min_e2 = torch.min(e2, dim=1).values
        # min_rel = torch.min(torch.cat((min_e1.unsqueeze(1), min_e2.unsqueeze(1)), dim=1), dim=1).values
        #
        # max_e1 = torch.max(e1, dim=1).values
        # max_e2 = torch.max(e2, dim=1).values
        # max_rel = torch.max(torch.cat((max_e1.unsqueeze(1), max_e2.unsqueeze(1)), dim=1), dim=1).values
        #
        # relation_spans = torch.cat((min_rel.unsqueeze(1), max_rel.unsqueeze(1)), dim=1)
        embeddings = self.embedder(tokens)

        mask = get_text_field_mask(tokens)

        if self.encoder is not None:

            embeddings = self.encoder(embeddings, mask)

        pooled = self.pooler(embeddings, mask)
        embeddings = embeddings[:, 1:, :]

        embeddings = self.dropout(embeddings)

        e1_mask = get_mask_for_spans(e1 - 1, embeddings.shape[:-1])
        e2_mask = get_mask_for_spans(e2 - 1, embeddings.shape[:-1])

        e1_span_embedding = self.extract_entity(embeddings, e1_mask)
        e2_span_embedding = self.extract_entity(embeddings, e2_mask)

        # e1_span_embedding = self.entity_extractor(embeddings, e1-1)
        # e2_span_embedding = self.entity_extractor(embeddings, e2-1)

        # relation_context_spans = torch.cat((e1[:, 1].unsqueeze(1), (e2[:, 1] - 1).unsqueeze(1)), dim=1)
        # relation_context_mask = get_mask_for_spans(relation_context_spans, embeddings.shape[:-1])
        # relation_context_embedding = self.extract_entity(embeddings, relation_context_mask)
        # pooled_output = torch.cat([e1_span_embedding, e2_span_embedding, pooled, relation_context_embedding], dim=-1)

        pooled_output = torch.cat([e1_span_embedding, e2_span_embedding, pooled], dim=-1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probabilities= self.softmax(logits)

        output = {"logits": logits, "probabilities": probabilities}

        if labels is not None:

            output["loss"] = self.loss(logits, labels)
            self.acc(logits, labels)
            self.f1(logits, labels)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1 = self.f1.get_metric(reset)
        # macro_f1 = sum(f1["fscore"]) / len(f1["fscore"])
        # result = {}
        #
        # for metric_name, metrics_per_class in f1.items():
        #     for class_index, value in enumerate(metrics_per_class):
        #         result[f"{self.label_tokens[class_index]}-{metric_name}"] = value
        # result["macro_f1"] = macro_f1

        result = f1
        result["acc"] = self.acc.get_metric(reset)

        return result

    def max_pooling(self, sequence, e_mask):
        entity_output = sequence * torch.stack([e_mask] * sequence.shape[-1], 2) + torch.stack(
            [(1.0 - e_mask) * -1000.0] * sequence.shape[-1], 2
        )
        entity_output = torch.max(entity_output, -2)[0]
        return entity_output.type_as(sequence)

    def extract_entity(self, sequence, e_mask):
        return self.max_pooling(sequence, e_mask)
