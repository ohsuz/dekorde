from typing import List, Tuple
from dekorde.paths import CONF_JSON,  GIBBERISH2KOR_TSV
import torch
import json
import csv


def load_gib2kor(mode) -> List[Tuple[str, str]]:
    TSV_PATH = GIBBERISH2KOR_TSV if mode == 'train' else GIBBERISH2KOR_INFER_TSV
    with open(TSV_PATH, 'r', encoding="utf-8") as fh:
        tsv_reader = csv.reader(fh, delimiter="\t")
        next(tsv_reader)  # skip the header
        return [
            (row[0], row[1])
            for row in tsv_reader
        ]


def load_conf() -> dict:
    with open(CONF_JSON, 'r', encoding="utf-8") as fh:
        conf = json.loads(fh.read())
    return conf


def load_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
