from typing import List, Tuple
import guidance
from guidance import models
from guidance import select
from tqdm import tqdm
from .datasets import DNLI, split_dialogue
from .config import config

ExampleLabelled = Tuple[str, str, str]
ExampleUnlabelled = Tuple[str, str]
ExampleDialogueLabelled = Tuple[List[Tuple[str, str]], str, str]
ExampleDialogueUnlabelled = Tuple[List[Tuple[str, str]], str]


def read_contents(fn: str) -> str:
    with open(fn, "r", encoding="utf-8") as file:
        contents = file.read()
    return contents


def read_examples(example_fn: str) -> List[ExampleLabelled]:
    with open(example_fn, "r", encoding="utf-8") as file:
        contents = [tuple(ln.strip().split('\t')) for ln in file.readlines()]
    return contents


def read_examples_dialogue(example_fn: str) -> List[ExampleDialogueLabelled]:
    with open(example_fn, "r", encoding="utf-8") as file:
        contents = [tuple(ln.strip().split('\t')) for ln in file.readlines()]
    return [(split_dialogue(dia), hyp, label) for (dia, hyp, label) in contents]


def get_model(server=True):
    if server:
        return models.Transformers(config['default-model'], **config['model-config'])
    else:
        return models.Transformers(config['default-model'], device=config['device'])


def prepare_prompts_nli_regular(prompt_instructions: str, examples: List[ExampleLabelled],
                                data: List[ExampleUnlabelled]) -> List[str]:
    example_template = "Premise: {}\nHypothesis: {}\nRelation: {}"
    prompt_template = "Premise: {}\nHypothesis: {}\nRelation: "
    formatted_examples = '\n'.join([example_template.format(prem, hyp, label) for (prem, hyp, label) in examples])
    prompts = [prompt_instructions + '\n' + formatted_examples + '\n' + prompt_template.format(prem, hyp)
               for prem, hyp in data]
    return prompts


def prepare_prompts_nli_dialogue(prompt_instructions: str, examples: List[ExampleDialogueLabelled],
                                data: List[ExampleDialogueUnlabelled]) -> List[str]:
    formatted_examples = '\n'.join(["\n".join(["Speaker {}: {}".format(speaker, text) for (speaker, text) in turns]) +
                                     "\nHypothesis: {}\nRelation: {}".format(hyp, label)
                          for (turns, hyp, label) in examples])
    formatted_prompts = ["\n".join(["Speaker {}: {}".format(speaker, text) for (speaker, text) in turns]) +
                          "\nHypothesis: {}\nRelation: ".format(hyp) for (turns, hyp) in data]
    prompts = [prompt_instructions + '\n' + formatted_examples + '\n' + p for p in formatted_prompts]
    return prompts


# The actual prompting for NLI
@guidance(stateless=True)
def nli_label(lm):
    return lm + select(["Entailment", "Contradiction", "Neutral"], name='label')


def prompt_model(model, prompt) -> str:
    lm = model + prompt + nli_label()
    return lm['label']


def main_test_regular(server=False):
    model = get_model(server=server)
    prompt_template = read_contents('./llm/prompts/nli_regular.txt')
    med_items = med.get_prem_hyp_pairs()
    examples = read_examples('./llm/prompts/examples1.txt')
    prompts = prepare_prompts_nli_regular(prompt_template, examples, med_items)
    labels = med.get_labels()
    preds = []
    for prompt in tqdm(prompts):
        preds.append(prompt_model(model, prompt))
    return preds, labels


def main_test_dialogue(server=False):
    model = get_model(server=server)
    prompt_template = read_contents('./llm/prompts/nli_dialogue.txt')
    dnli = DNLI('./data/compiled/data.csv')
    dnli_items = dnli.get_dialogue_hyp_pairs()
    examples = read_examples_dialogue('./llm/prompts/examples2.txt')
    prompts = prepare_prompts_nli_dialogue(prompt_template, examples, dnli_items)
    labels = dnli.get_labels()
    preds = []
    for prompt in tqdm(prompts):
        try:
            preds.append(prompt_model(model, prompt))
        except AssertionError:
            print(prompt)
    return preds, labels


def test():
    dnli = DNLI('./data/compiled/data.csv')
    dnli_items = dnli.get_dialogue_hyp_pairs()

