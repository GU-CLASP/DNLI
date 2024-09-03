# Define datasets etc.
import csv
from typing import NamedTuple, List, Tuple

UP = 'UP'
DOWN = 'DOWN'
NON = 'NON'


class CompactSample(NamedTuple):
    index: int
    premise: str
    hypothesis: str
    label: str
    mono: str
    features: List[str]


def get_monotonicity(props: str) -> str:
    if 'upward_monotone' in props:
        return UP
    elif 'downward_monotone' in props:
        return DOWN
    elif 'non_monotone' in props:
        return NON
    else:
        raise ValueError("No monotonicity found!")


def get_features(props: str) -> List[str]:
    non_features = ['upward_monotone', 'downward_monotone', 'non_monotone', 'GLUE', 'FraCaS_GQ', 'crowd', 'paper']
    features = list(filter(lambda p: p not in non_features, props.split(':')))
    if features:
        return list(map(lambda s: s.lower(), features))
    else:
        return ['other']


class MED(object):
    def __init__(self, med_fn: str):
        self.med_fn = med_fn
        self.name = self.med_fn.split('/')[-1].split('.')[0]
        self.data = self.load_data()

    def load_data(self):
        with open(self.med_fn, 'r') as in_file:
            lines = [ln.strip().split('\t') for ln in in_file.readlines()][1:]
        return [CompactSample(int(ln[0]), ln[8], ln[9], ln[15],
                              get_monotonicity(ln[3]), get_features(ln[3])) for ln in lines]

    def get_sentences(self):
        premises = [d.premise for d in self.data]
        hypotheses = [d.hypothesis for d in self.data]
        return list(set(premises+hypotheses))

    def get_prem_hyp_pairs(self):
        return [(d.premise, d.hypothesis) for d in self.data]

    def get_labels(self):
        return [d.label for d in self.data]

class CompactDialogueSample(NamedTuple):
    index: int
    dialogue: List[Tuple[str, str]]
    hypothesis: str
    label: str

def split_dialogue(line: str) -> List[Tuple[str, str]]:
    turns = [t.split('>') for t in line.split('<')[1:]]
    return [(speaker, text.strip()) for (speaker, text) in turns]

class DNLI(object):
    def __init__(self, dnli_fn: str):
        self.dnli_fn = dnli_fn
        self.name = 'DNLI'
        self.data = self.load_data()

    def load_data(self) -> List[CompactDialogueSample]:
        with open(self.dnli_fn, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            items = [CompactDialogueSample(i, split_dialogue(row[0]), row[1], row[2]) for i, row in enumerate(reader)]
        return items

    def get_dialogue_hyp_pairs(self):
        return [(d.dialogue, d.hypothesis) for d in self.data]

    def get_labels(self):
        return [d.label for d in self.data]