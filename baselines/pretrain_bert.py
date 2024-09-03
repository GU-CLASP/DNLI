import transformers
import torch
from pathlib import Path
from torch.utils.data import Dataset
import csv
from argparse import ArgumentParser
import os

arg_parser = ArgumentParser()
arg_parser.add_argument('--gpu-id', default=None)
arg_parser.add_argument('--load-pretrained', dest='load_pretrained', action='store_true')
arg_parser.add_argument('--no-load-pretrained', dest='load_pretrained', action='store_false')
arg_parser.set_defaults(load_pretrained=False)
args = arg_parser.parse_args()

class BNCDataset(Dataset):
    def __init__(self, split, tokenizer=None):
        self.tokenizer = tokenizer
        self.examples = []
        if split == 'train':
            corpus_dir = Path('data/pretraining/')
        elif split == 'dev':
            corpus_dir = Path('data/source/dev/')
        else:
            raise ValueError(f"No split named {split}")
        src_files = corpus_dir.glob("*.csv") 
        for src_file in src_files:
            with open(src_file) as f:
                reader = csv.reader(f, delimiter='\t')
                for line in list(reader)[:10]:
                    speaker = line[0]
                    if speaker == '[ANNOTATION]':
                        continue
                    utt = line[1]
                    utt = self.tokenizer.encode(utt) if self.tokenizer else utt
                    self.examples.append(utt) 
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, i):
        return self.examples[i]

os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu_id}'

if args.load_pretrained:
    print("Loading pretrained BERT...")
    bert = transformers.BertForMaskedLM.from_pretrained('bert-base-uncased')
    model_path = 'models/bert-pretraining+bnc-pretraining'
else:
    print("Training BERT from scratch...")
    bert_config = transformers.BertConfig()
    bert = transformers.BertForMaskedLM(bert_config)
    model_path = 'models/bnc-pretraining'

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

train_data = BNCDataset('train', tokenizer)
dev_data = BNCDataset('dev', tokenizer)

data_collator = transformers.DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = transformers.TrainingArguments(
    output_dir=model_path,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=10,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=64,
    logging_steps=100,
    load_best_model_at_end=True,
    save_total_limit=5,
)

trainer = transformers.Trainer(
    model=bert,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=dev_data,
)

trainer.train()


