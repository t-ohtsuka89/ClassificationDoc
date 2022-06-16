from dataclasses import dataclass
from enum import IntEnum


@dataclass
class SpecialToken(IntEnum):
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3
    CLS = 4
