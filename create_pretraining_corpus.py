from argparse import ArgumentParser
from pathlib import Path
import xml.etree.ElementTree as ET
import string
import csv

arg_parser = ArgumentParser()
arg_parser.add_argument('bnc_corpus_dir')
args = arg_parser.parse_args()

def read_bnc_file_untagged(path):
    speaker_names = iter(string.ascii_uppercase)
    speakers = {}
    speakers['UNKFEMALE'] = 'unknown_female'
    speakers['UNKMALE'] = 'unknown_female'
    root = ET.parse(path).getroot()
    for t in root.findall('body/u'):
        text = t.text.strip() if t.text else None
        speaker = t.attrib['who']
        if not speaker in speakers:
            speakers[speaker] = next(speaker_names)
        if text:
            yield speakers[speaker], text

def read_bnc_file_tagged(path):
    speaker_names = iter(string.ascii_uppercase)
    speakers = {}
    speakers['UNKFEMALE'] = 'unknown_female'
    speakers['UNKMALE'] = 'unknown_female'
    root = ET.parse(path).getroot()
    for u in root.findall('u'): # utterances
        speaker_id = u.attrib['who']
        if not speaker_id in speakers:
            speakers[speaker_id] = next(speaker_names)
        speaker = speakers[speaker_id].strip()
        text = ' '.join([w.text for w in u.findall('w')]).strip()
        if text:
            yield speaker, text

if __name__ == '__main__':

    dnli_corpus_dir =  Path("./data/source")
    dnli_corpus_files = []
    for split in ['train', 'test', 'dev']:
        for path in (dnli_corpus_dir/split).glob('*.csv'):
            dnli_corpus_files.append(path.stem)
    bnc_files_used = [f[4:] for f in dnli_corpus_files if f.startswith('BNC_')]

    bnc_corpus_dir = Path(args.bnc_corpus_dir)
    # bnc_corpus_dir = Path("../DialogueNLI/data/pilot2_prompts/corpora/BNC2014/spoken")
    pretrain_corpus_files = []

    for bnc_file in (bnc_corpus_dir/'untagged').glob("*.xml"):
        with open(f'data/pretraining/BNC_{bnc_file.stem}.csv', 'w') as f:
            writer = csv.writer(f,  delimiter='\t')
            for line in read_bnc_file_untagged(bnc_file):
                    writer.writerow(line)

