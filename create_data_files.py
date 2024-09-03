import os
import random
from IPython.terminal.embed import embed
import spacy
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")
tokenize = False

def format_context(context):
    context_str = ''
    for s, u in context:
        if tokenize:
            u = ' '.join([x.text for x in nlp(u)])
        context_str += f'<{s}> {u.lower()} '
    return context_str

def format_data(data_split, num_previous_turns):
    out = open(f'data/compiled/{data_split}_{num_previous_turns}_data_t={tokenize}.csv', '+w')
    ddir = f'data/source/{data_split}/'
    
    for k, fname in enumerate(os.listdir(ddir)):
        examples = read_file(ddir+fname, num_previous_turns)
        for line in examples:
            out.write(line)
        
def read_file(path, num_previous_turns):
    examples = []
    with open(path) as f:
        all_lines = []
        for i, line in enumerate(f):
            #print(i, end='\r')
            if line.startswith('['):
                _, hyp, label = line.rstrip().split('\t')
                context = all_lines[-num_previous_turns:]
                context_str = format_context(context)
                if tokenize:
                    hyp = ' '.join([x.text for x in nlp(hyp)])
                hyp = hyp.lower()
                examples.append('\t'.join([context_str, hyp, label])+'\n')
            else:
                speaker, utterance = line.rstrip().split('\t')
                all_lines.append((speaker, utterance))
    return examples

def n_context_all_splits(n=5):
    print('generating train...')
    format_data('train', n)
    
    print('generating dev...')
    format_data('dev', n)
    
    print('generating test...')
    format_data('test', n)
    
    print()
    
def ood_data(n=5):
    """
    bnc as training, childes as test
    """
    train_dev = []
    test = []
    
    ddir = f'data/source/'
    
    for k, split in enumerate(os.listdir(ddir)):
        new_path = os.path.join(ddir, split)
        for fname in os.listdir(new_path):
            examples = read_file(new_path+'/'+fname, n)
            if 'BNC' in fname:
                train_dev.append(examples)
            else:
                test.append(examples)
    
    dev, train = [], []
    for file_examples in train_dev:
        for example in file_examples:
            if random.random() > 0.8:
                dev.append(example)
            else:
                train.append(example)
    
    for name, split in [('train_bnc', train), ('dev_bnc', dev), ('test_childes', test)]:
        with open(f'data/compiled/ood_{name}_{tokenize}.csv', '+w') as f:
            for file_examples in split:
                for example in file_examples:
                    f.write(example)
    
if __name__ == '__main__':
    for n in [1, 3, 5, 7, 9, 11, 13, 15]:
        n_context_all_splits(n)
    
    #ood_data()