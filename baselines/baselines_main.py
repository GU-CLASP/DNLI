import random

from IPython.terminal.embed import embed

from data import data_batcher, dataloader
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import numpy as np
from models import DialogueFlatConcat, DialogueHierarchical, HypOnlyModel, TransformerModel
from args import args
from transformers.optimization import AdamW
from transformers import logging

logging.set_verbosity_error() # errors and critical warnings

DEVICE = torch.device(args.device)

def model_initialization(model):
    for w in model.parameters():
        if len(w.size()) > 1:
            #nn.init.orthogonal_(w.data)
            nn.init.kaiming_normal_(w.data)
            
def save_model(model, e):
    torch.save(model.state_dict(), f'./models/dnli_bert={args.model_configuration}.pt')

def main(n):
    
    if 'bert' in args.model_configuration or 'transformer' in args.model_configuration:
        t = False
    else:
        t = True
        
    train_dataset, w2i, r2i = dataloader(f'data/compiled/train_{n}_data_t={t}.csv')
    dev_dataset, *_ = dataloader(f'data/compiled/dev_{n}_data_t={t}.csv')
    test_dataset, *_ = dataloader(f'data/compiled/test_{n}_data_t={t}.csv')

    print(len(train_dataset))
    
    if 'hyp_only' in args.model_configuration:
        model = HypOnlyModel(len(w2i), len(r2i))
    elif 'transformer' in args.model_configuration:
        model = TransformerModel(len(r2i))
    elif 'hierarchical' in args.model_configuration:
        model = DialogueHierarchical(len(w2i), len(r2i))
    else:
        model = DialogueFlatConcat(len(w2i), len(r2i))
        
    model.to(DEVICE)
    #num_params = sum([p.numel() for p in model.parameters()])

    if 'bert' in args.model_configuration:
        if 'hierarchical' in args.model_configuration:
            param_list = [{'params': model.hyp_encoder.parameters(), 'lr': args.bert_lr},
                          {'params': model.classifier.parameters(), 'lr': args.lr},
                          {'params': model.embs.parameters(), 'lr': args.lr},
                          {'params': model.turn_lstm.parameters(), 'lr': args.lr},
                          {'params': model.utt_lstm.parameters(), 'lr': args.lr}]
        else:
            if 'only' in args.model_configuration:
                param_list = [{'params': model.hyp_encoder.parameters(), 'lr': args.bert_lr},
                              {'params': model.classifier.parameters(), 'lr': args.lr}]
            else:
                param_list = [{'params': model.hyp_encoder.parameters(), 'lr': args.bert_lr},
                              {'params': model.classifier.parameters(), 'lr': args.lr},
                              {'params': model.embs.parameters(), 'lr': args.lr},
                              {'params': model.lstm.parameters(), 'lr': args.lr}]
            
        if 'hierarchical' in args.model_configuration:
            ### set parameters if attention modules are used
            if args.turn_pooling == 'self_att':
                param_list += [{'params':model.att_w.parameters(), 'lr':args.lr}]
            elif args.turn_pooling == 'att':    
                param_list += [{'params':model.reduce_seq_turn.parameters(), 'lr':args.lr}]
            else:
                pass
        else:
            pass
        
        if args.token_pooling == 'self_att' and 'only' not in args.model_configuration:
            param_list += [{'params':model.att_w_token.parameters(), 'lr':args.lr}]
        elif args.token_pooling == 'att' and 'only' not in args.model_configuration:    
            param_list += [{'params':model.reduce_seq_token.parameters(), 'lr':args.lr}]
        else:
            pass
        
        opt = AdamW(param_list, weight_decay=args.weight_decay)
    elif 'transformer' in args.model_configuration:
        opt = AdamW(model.parameters(), lr=args.bert_lr, weight_decay=args.weight_decay)
    else:
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_dev = 0.
    best_e = 0
    for e in range(args.n_epochs):
        # might be better: 
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, 
                                                         T_max=int(len(train_dataset)/args.batch_size)+1, 
                                                         eta_min=args.min_lr)
        
        train_iter = data_batcher(train_dataset, w2i, r2i, batch_size=args.batch_size)
        tr_loss, tr_acc = train(e, model, train_iter, opt, scheduler)
    
        dev_iter = data_batcher(dev_dataset, w2i, r2i, batch_size=args.batch_size)
        dev_loss, dev_acc = validate(model, dev_iter)
        
        if args.train_verbose:
            print(f'{e} ::: {tr_loss:.3f}, {tr_acc:.3f} | {dev_loss:.3f}, {dev_acc:.3f} | save={dev_acc > best_dev}')
        
        if e == 0:
            save_model(model, e)
        
        if dev_acc > best_dev:
            save_model(model, e+1)
            best_dev = dev_acc
            best_e = e+1
            
    model.load_state_dict(
        torch.load(f'./models/dnli_bert={args.model_configuration}.pt'))
    
    test_iter = data_batcher(test_dataset, w2i, r2i)
    loss, acc = validate(model, test_iter)
    
    #print('TEST:::', n, loss, acc)
    #print()
    return acc

def train(e, model, data_iter, opt, scheduler):
    model.train()
    acc = []
    losses = []
    for j, batch in enumerate(data_iter):

        if 'bert' in args.model_configuration:
            h = torch.tensor(batch.hypothesis['input_ids'], device=DEVICE).long()
            h_mask = torch.tensor(batch.hypothesis['attention_mask'], device=DEVICE).long()
        else:
            h = torch.tensor(batch.hypothesis, device=DEVICE).long()
            h_mask = None
            
        if 'transformer' in args.model_configuration:
            p = torch.tensor(batch.premise['input_ids'], device=DEVICE).long()
            p_mask = torch.tensor(batch.premise['attention_mask'], device=DEVICE).long()
        else:
            p = torch.tensor(batch.premise, device=DEVICE).long()
            p_mask = (p != 0)
            
        gold = torch.tensor(batch.label, device=DEVICE).long()
        
        if 'hierarchical' in args.model_configuration or 'transformer' in args.model_configuration:
            #speakers = torch.tensor(batch.speakers, device=DEVICE).long()
            output = model(h, p, p_mask=p_mask, h_mask=h_mask)
        else:
            if 'bert' in args.model_configuration:
                output = model(h, p, p_mask=p_mask, h_mask=h_mask)
            else:
                output = model(h, p, p_mask=p_mask)
        
        loss = F.cross_entropy(output, gold.squeeze())
        losses += [loss.item()]
            
        acc += (torch.argmax(output, 1) == gold.squeeze()).tolist()
        
        loss.backward()
        clip_grad_norm_(model.parameters(), args.clipping)
        opt.step()
        scheduler.step()
        opt.zero_grad()
        
    return np.round(np.mean(losses),5), np.round(np.mean(acc),3)
        
def validate(model, data_iter):
    model.eval()
    acc = []
    losses = []
    for batch in data_iter:
        
        if 'bert' in args.model_configuration:
            h = torch.tensor(batch.hypothesis['input_ids'], device=DEVICE).long()
            h_mask = torch.tensor(batch.hypothesis['attention_mask'], device=DEVICE).long()
        else:
            h = torch.tensor(batch.hypothesis, device=DEVICE).long()
            h_mask = (h != 0)
        
        if 'transformer' in args.model_configuration:
            p = torch.tensor(batch.premise['input_ids'], device=DEVICE).long()
            p_mask = torch.tensor(batch.premise['attention_mask'], device=DEVICE).long()
        else:
            p = torch.tensor(batch.premise, device=DEVICE).long()
            p_mask = (p != 0)
            
        gold = torch.tensor(batch.label, device=DEVICE).long()
        
        with torch.no_grad():
            if 'hierarchical' in args.model_configuration or 'transformer' in args.model_configuration:
                #speakers = torch.tensor(batch.speakers, device=DEVICE).long()         
                output = model(h, p, p_mask=p_mask, h_mask=h_mask)
            else:
                if 'bert' in args.model_configuration:
                    output = model(h, p, p_mask=p_mask, h_mask=h_mask)
                else:
                    output = model(h, p, p_mask=p_mask)
        
        loss = F.cross_entropy(output, gold.squeeze())
        losses += [loss.item()]
        acc += (torch.argmax(output, 1) == gold.squeeze()).tolist()
        #print(np.round(np.mean(losses), 5), np.round(np.mean(acc), 3), end='\r')
        
    return np.round(np.mean(losses), 5), np.round(np.mean(acc), 3)

def context_experiments(num_runs=3):
    args.train_verbose = False
    
    for n in [3, 5, 7, 9, 11, 13, 15]:
        n_accs = []
        for _ in range(num_runs):
            torch.cuda.empty_cache()
            rseed = random.randint(0,1000)
            torch.manual_seed(rseed)
            
            acc = main(n)
            print(f'>>> {acc}')
            n_accs.append(acc)

        print(f'>>>>> context = {n}, mean acc = {np.mean(n_accs)}, std = {np.std(n_accs)}')

def baselines(num_runs=3):
    args.train_verbose = False
    for model in ['hyp_only_bert', 'hyp_only_lstm']:
        args.model_confiiguration = model
        n_accs = []
        for _ in range(num_runs):
            torch.cuda.empty_cache()
            rseed = random.randint(0,1000)
            torch.manual_seed(rseed)
            
            acc = main(1)
            print(f'>>> {acc}')
            n_accs.append(acc)

        print(f'>>>>> context = {model}, mean acc = {np.mean(n_accs)}, std = {np.std(n_accs)}')

if __name__ == '__main__':
    # context_experiments()
    
    # 'lstm_flat', 'lstm_hierarchical', 
    # for mc in ['bert_flat', 'bert_hierarchical']:#, 'hierarchical_transformer', 'flat_transformer']:
    #     args.model_configuration = mc
    #     print(f'\n========= {mc} =========')
    #     context_experiments()
    
    baselines()
    
    # torch.cuda.empty_cache()
    # torch.manual_seed(333)
    # acc = main(5)
    # print(acc)
