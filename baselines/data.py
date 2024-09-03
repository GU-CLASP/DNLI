from collections import defaultdict
import random
from toolz import take
from collections import defaultdict, namedtuple
import numpy as np
from IPython import embed
from transformers import BertTokenizer, AutoTokenizer
import re
import string
from copy import deepcopy
from args import args
from itertools import chain

if 'bert' in args.model_configuration or 'transformer' in args.model_configuration:
    bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
    dialogue_tok = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    dialogue_tok.add_special_tokens({'pad_token': '[PAD]'})
    
letter_map = {c:i for i, c in enumerate(string.ascii_uppercase)}

def dataloader(path):
    dataset = []
    w2i = defaultdict(int)
    r2i = defaultdict(int)
    
    w2i['<pad>'] = 0
    w2i['<unk>'] = 1
        
    with open(path) as f:
        for line in f:
            premise, hyp, label = line.rstrip().split('\t')
            premise = combine_special_tags(tokenize(premise))
            hyp = tokenize(hyp)
            if label not in r2i:
                r2i[label] = len(r2i)
            
            if 'bert' not in args.model_configuration:
                vocab_source = premise + hyp
            else:
                vocab_source = premise
                
            for w in vocab_source:
                if w not in w2i:
                    w2i[w] = len(w2i)
                
            dataset.append([premise, hyp, label])

    return dataset, w2i, r2i

def numericalize(seqs, vocab):
    return [[vocab.get(w, 1) for w in x] for x in seqs]

def pad(seqs, pad_idx, maxl=False):
    if maxl == False:
        maxl = max([len(x) for x in seqs])
    return np.array([x+[pad_idx]*(maxl-len(x)) for x in seqs])

def combine_special_tags(text):
    text = ' '.join(text)
    speaker_tags = re.findall('(< ([A-Z]) >)', text)
    for speaker_tag, ch in set(speaker_tags):
        text = re.sub(speaker_tag, f'<{ch}>', text)
    return text.split()        
    
def tokenize(text):
    return text.split()
    #return [x.text for x in nlp(text)]

def data_batcher(dataset, w2i, r2i, batch_size = 8):
    random.shuffle(dataset)
    dataset_size = len(dataset)
    dataset = iter(dataset)
    Batch = namedtuple('Batch', ['premise', 'hypothesis', 'label'])
    
    for _ in range(int(dataset_size/batch_size)+1):
        batch = list(take(batch_size, dataset))
        premise, hypothesis, labels = list(zip(*batch))
        
        if 'transformer' in args.model_configuration:
            if 'hierarchical' in args.model_configuration:
                _, turns = split_turns(premise)
                max_tokens = max([max([len(s.split(' ')) for s in x]) for x in turns])
                
                premise = {'input_ids':[], 'attention_mask':[]}
                
                for i in range(len(turns)):
                    premise_dick = dialogue_tok.batch_encode_plus(turns[i],
                                                                  padding='max_length',
                                                                  max_length=max_tokens, 
                                                                  truncation=True)
                    premise['input_ids'].append(premise_dick['input_ids'])
                    premise['attention_mask'].append(premise_dick['attention_mask'])

            else:
                turns = [re.sub('(<[A-Za-z_]*>|unknown_female|unknown_male)', '', ' '.join(x)).strip() for x in premise]
                premise = dialogue_tok.batch_encode_plus(turns, 
                                                         padding=True, 
                                                         truncation=True)
        elif 'hierarchical' in args.model_configuration:
            speakers, turns = split_turns(premise)
            turns = pad_turns(turns)
            premise = list_of_turns(turns, w2i)
            speakers = format_speakers(speakers)  
        else:
            premise = pad(numericalize(premise, w2i), w2i['<pad>'])
            
        if 'bert' in args.model_configuration:
            hypothesis = bert_tok.batch_encode_plus([' '.join(x).lower() for x in hypothesis], 
                                                     padding=True, 
                                                     truncation=True)
        else:
            hypothesis = pad(numericalize(hypothesis, w2i), w2i['<pad>'])
            
        labels = np.array(numericalize([[x] for x in labels], r2i))
    
        yield Batch(premise, hypothesis, labels) 
        
def pad_turns(turns):
    _turns = deepcopy(turns)
    for i, dialogue in enumerate(_turns):
        if len(dialogue) < 5:
            turns[i] += ['<pad>'] * (5 - len(dialogue))
    return turns

def format_speakers(speakers):
    batch_speakers = []
    for dialogue_speakers in speakers:
        batch_speakers.append([])
        dialogue_map_ = {v:k for k,v in enumerate(set(dialogue_speakers))}
        for speaker in dialogue_speakers:
            batch_speakers[-1].append(dialogue_map_[speaker])
        
        # speaker padding
        if len(batch_speakers[-1]) < 5:
            batch_speakers[-1] += [0]*(5-len(dialogue_speakers))

    return np.array(batch_speakers)

def list_of_turns(turns, w2i):
    max_turns = max([len(x) for x in turns]) # max num of turns
    max_tokens = max([max([len(s.split(' ')) for s in x]) for x in turns]) # max num of words
    turns_tensor = np.zeros((len(turns), max_turns, max_tokens))
    
    for i in range(max_turns):
        xss = []
        for x in turns:
            try:
                xss.append(x[i].split())
            except:
                xss.append(['padded_turn'])
        try:
            turn_i = pad(numericalize(xss, w2i), w2i['<pad>'], max_tokens)
            turns_tensor[:,i,:] = np.array(turn_i)
        except:
            print('data error...')
            embed()
            assert False

    return turns_tensor

def remove_lr_space(x):
    if x.startswith(' '):
        x = x[1:]
    if x.endswith(' '):
        x = x[:-1]
    return x
        
def split_turns(context):
    speakers = [re.findall(r'<[A-Z]>', ' '.join(x)) for x in context]
    speakers = [[y.replace('>','').replace('<','') for y in x] for x in speakers]
    turns = [[remove_lr_space(z) for z in re.split(r'<[A-Za-z_]*>', ' '.join(x))[1:]] for x in context]
    return speakers, turns
    
if __name__ == '__main__':
    data, w2i, r2i = dataloader('data.csv')
    batcher = data_batcher(data, w2i, r2i, data_type='hierarchical')
    x = next(batcher)
    embed()
