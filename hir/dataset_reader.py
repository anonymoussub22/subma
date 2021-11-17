"""
This module reads all used text datasets and translates them to allennlp Instances
"""
import csv
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Set, Tuple

from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import (
    LabelField,
    ListField,
    MetadataField,
    SequenceLabelField,
    SpanField,
    TensorField,
    TextField,
)
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer


def get_element_spans(tokens: List[str], start_offset=0) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    e11_p = tokens.index("<e1>")  # the start position of entity1
    e12_p = tokens.index("</e1>")  # the end position of entity1
    e21_p = tokens.index("<e2>")  # the start position of entity2
    e22_p = tokens.index("</e2>")  # the end position of entity2
    return (e11_p + start_offset, e12_p - 2 + start_offset), (e21_p - 2 + start_offset, e22_p - 4 + start_offset)


def read_csv(path: Path) -> Generator[str, None, None]:
    with path.open("r") as text_line:
        for line in csv.reader(text_line, delimiter="\t", strict=True, quotechar=None):
            yield line


@DatasetReader.register("pre-parsed")
class PreParsedReader(DatasetReader):
    """
    This class reads pre-parsed sentences
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()

        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(
        self,
        text: str,
        label: str = None,
        rediction: Set[int] = None,
    ) -> Instance:

        tokens = text.strip().split(" ")

        if rediction is not None and isinstance(rediction, list):

            reduced_tokens = []
            i = 0
            for token in tokens:
                if token in ["<e1>", "</e1>", "<e2>", "</e2>"]:
                    reduced_tokens += [token]
                    continue

                if i not in rediction:
                    reduced_tokens += [token]
                i += 1
            tokens = reduced_tokens

        elif rediction is not None and rediction == "complete":
            reading_state = "NO_ENT"
            reduced_tokens = []
            for token in tokens:
                if token in ["<e1>", "<e2>"]:
                    reading_state = "IN_ENT"
                    reduced_tokens += [token]
                    continue
                if token in ["</e1>", "</e2>"]:
                    reading_state = "NO_ENT"
                    reduced_tokens += [token]
                    continue

                if reading_state == "IN_ENT":
                    reduced_tokens += [token]

            tokens = reduced_tokens + ["."]

        elif rediction is not None and rediction == "keep_inner":
            reading_state = "OUTER"
            reduced_tokens = []
            for token in tokens:
                if token in ["<e1>"]:
                    reading_state = "INNER"
                    reduced_tokens += [token]
                    continue
                if token in ["</e2>"]:
                    reading_state = "OUTER"
                    reduced_tokens += [token]
                    continue

                if reading_state == "INNER":
                    reduced_tokens += [token]

            tokens = reduced_tokens + ["."]

        elif rediction is not None and rediction == "keep_outer":
            reading_state = "OUTER"
            reduced_tokens = []
            for token in tokens:
                if token in ["</e1>"]:
                    reading_state = "INNER"
                    reduced_tokens += [token]
                    continue
                if token in ["<e2>"]:
                    reading_state = "OUTER"
                    reduced_tokens += [token]
                    continue

                if reading_state == "OUTER":
                    reduced_tokens += [token]

            tokens = reduced_tokens + ["."]

        words = [token for token in tokens if token not in ["<e1>", "</e1>", "<e2>", "</e2>"]]

        text = TextField([Token("[CLS]")] + [Token(w) for w in words], self._token_indexers)

        e1_idx, e2_idx = get_element_spans(tokens, start_offset=1)

        e1_span_field = SpanField(*e1_idx, text)
        e2_span_field = SpanField(*e2_idx, text)

        fields = {"tokens": text, "e1": e1_span_field, "e2": e2_span_field, "metadata": MetadataField({})}
        if label is not None:
            fields["labels"] = LabelField(label, label_namespace="labels")

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        text_fp = Path(file_path)

        csv_reader = read_csv(text_fp)

        for e1, e2, label, text in csv_reader:  # pylint: disable=unused-variable

            yield self.text_to_instance(
                text=text,
                label=label,
            )
