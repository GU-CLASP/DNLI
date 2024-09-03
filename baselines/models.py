from numpy.core.fromnumeric import argmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from transformers.models.bert.tokenization_bert import BertTokenizer
from args import args
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BertModel

DEVICE = torch.device(args.device)

class TransformerModel(nn.Module):
    def __init__(self, labels) -> None:
        super(TransformerModel, self).__init__()
        self.hyp_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.dialogue_encoder = AutoModel.from_pretrained("microsoft/DialoGPT-small")
        
        if 'hierarchical' in args.model_configuration:
            self.lstm = nn.LSTM(768, 768, bidirectional=False, batch_first=True) 
        
        self.dialogue_encoder.resize_token_embeddings(50257+1)
        
        self.combine_h_p = lambda h, p: torch.cat([p, h, abs(p-h), p*h], -1)
        self.classifier = nn.Linear(3072, labels)
        
    def forward(self, h, p, p_mask=None, h_mask=None):
        h_e = self.hyp_encoder(input_ids=h, 
                               attention_mask=h_mask).pooler_output
        
        if 'hierarchical' in args.model_configuration:
            dialogue_turns = torch.zeros((p.size(0), p.size(1), 768), device=DEVICE)
            if len(p.size()) == 2: # wierd scenario when seql = 1
                p = p.unsqueeze(1)
                
            for i in range(p.size(1)):
                s_dialogues = self.dialogue_encoder(p[:,i,:]).last_hidden_state
                dialogue_turns[:,i,:] = s_dialogues.max(1).values

            dialogues, *_ = self.lstm(dialogue_turns)
            dialogues = dialogues.max(1).values
        else:
            dialogues = self.dialogue_encoder(p).last_hidden_state                
            dialogues = dialogues.max(1).values
            
        #dialogues = self.dialogue_pooler(dialogues)        
        ph = self.combine_h_p(h_e, dialogues)
        
        return self.classifier(ph)

class DialogueFlatConcat(nn.Module):
    def __init__(self, vocab, labels):
        super(DialogueFlatConcat, self).__init__()
        self.embs = nn.Embedding(vocab, 256, padding_idx=0)
        self.lstm = nn.LSTM(256, 384, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(3072, labels)
        
        self.dropout = nn.Dropout(0.4)
        if 'bert' in args.model_configuration:
            self.hyp_encoder = AutoModel.from_pretrained('bert-base-uncased') # add bert here
            #self.projection = 
            
        self.combine_h_p = lambda h, p: torch.cat([p, h, abs(p-h), p*h], -1)
        self.reduce_seq = lambda x: torch.max(x, 1).values
        
        ### token pooling
        if args.token_pooling == 'self_att':
            self.att_w_token = nn.Linear(768, 768)
            self.reduce_seq_token = lambda x: (F.softmax(self.att_w_token(x), -1)*x).sum(1)
        elif args.token_pooling == 'att':
            self.reduce_seq_token = BahdanauAttention(512, 512)
        else:
            self.reduce_seq_token = lambda x: torch.max(x, 1).values
        
    def forward(self, h, p, p_mask=None, h_mask=None):

        if 'bert' in args.model_configuration:
            h_e = self.hyp_encoder(input_ids=h, 
                                   attention_mask=h_mask).pooler_output
            #h_e = self.projection(h_e)
        else:
            h_e = self.embs(h)
            h_e, *_ = self.lstm(h_e)
            
        p_e = self.embs(p)
        p_e, *_ = self.lstm(p_e)
        
        h_e = self.dropout(h_e)
        p_e = self.dropout(p_e)
        
        # reduce from (B, S, D) to (B, D)
        if args.token_pooling == 'att':          
            p_e = self.reduce_seq_token(h_e, p_e)
        else:
            if 'bert' not in args.model_configuration:
                h_e = self.reduce_seq_token(h_e)
            p_e = self.reduce_seq_token(p_e)

        ph = self.combine_h_p(h_e, p_e)
        ph = self.dropout(ph)
        
        return self.classifier(ph)
    
class DialogueHierarchical(nn.Module):
    def __init__(self, vocab, labels):
        super(DialogueHierarchical, self).__init__()
        self.embs = nn.Embedding(vocab, 256, padding_idx=0)
        
        self.utt_lstm = nn.LSTM(256, 256, bidirectional=True, batch_first=True)
        self.turn_lstm = nn.LSTM(512, 384, bidirectional=True, batch_first=True)
        
        if 'bert' in args.model_configuration:
            self.hyp_encoder = BertModel.from_pretrained('bert-base-uncased') # add bert here
        else:
            self.projection = nn.Linear(512, 768)
        
        self.combine_h_p = lambda h, p: torch.cat([p, h, abs(p-h), p*h], -1)
        self.classifier = nn.Linear(3072, labels)
        
        self.dropout_p = nn.Dropout(0.4)
        self.dropout_h = nn.Dropout(0.5)
        
        ### token pooling
        if args.token_pooling == 'self_att':
            self.att_w_token = nn.Linear(512, 512)
            self.reduce_seq_token = lambda x: (F.softmax(self.att_w_token(x), -1)*x).sum(1)
        elif args.token_pooling == 'att':
            self.reduce_seq_token = BahdanauAttention(512, 512)
        else:
            self.reduce_seq_token = lambda x: torch.max(x, 1).values
        
        ### turn pooling
        if args.turn_pooling == 'self_att':
            self.att_w = nn.Linear(768, 768)
            self.reduce_seq_turn = lambda x: (F.softmax(self.att_w(x), -1)*x).sum(1)
        elif args.turn_pooling == 'att':    
            self.reduce_seq_turn = BahdanauAttention(768, 768)
        else:
            self.reduce_seq_turn = lambda x: torch.max(x, 1).values
        
    def forward(self, h, p, p_mask=None, h_mask=None):
        p_e = self.compute_utt_representations(p)
        p_e, (p_h, p_c) = self.turn_lstm(p_e)
            
        if 'bert' in args.model_configuration:
            h_e = self.hyp_encoder(input_ids=h, 
                                   attention_mask=h_mask).pooler_output
        else:
            h_e = self.embs(h)
            h_e, *_ = self.utt_lstm(h_e)
            h_e = torch.max(self.projection(h_e), 1).values
        
        p_e = self.dropout_p(p_e)
        h_e = self.dropout_h(h_e)

        # att
        if args.turn_pooling == 'att':
            p_e = self.reduce_seq_turn(h_e, p_e)
        else:
            p_e = self.reduce_seq_turn(p_e)
        
        ph = self.combine_h_p(h_e, p_e)
        ph = self.dropout_h(ph)
        
        return self.classifier(ph)
    
    def compute_utt_representations(self, p, h=None):
        turn_repr = torch.zeros(p.size(0), p.size(1), 512).to(DEVICE)
        p = self.embs(p)
        
        for i in range(p.size(1)):
            utt_i = p[:,i,:]
            utt_r, (h, c) = self.utt_lstm(utt_i)
            if args.token_pooling == 'att':
                assert 'Not Implemented yet bitch'
            else:
                utt_r = self.reduce_seq_token(utt_r)
            turn_repr[:,i,:] = utt_r
        
        return turn_repr
    
class HypOnlyModel(nn.Module):
    def __init__(self, vocab, labels):
        super(HypOnlyModel, self).__init__()
        if 'bert' in args.model_configuration:
            self.hyp_encoder = BertModel.from_pretrained('bert-base-uncased')
            self.classifier = nn.Linear(768, labels)
        else:
            self.embs = nn.Embedding(vocab, 256)
            self.hyp_encoder = nn.LSTM(256, 256, bidirectional=True, batch_first=True)
            self.classifier = nn.Linear(512, labels)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, h, p, p_mask=None, h_mask=None):
        
        if 'bert' in args.model_configuration:
            h_e = self.hyp_encoder(input_ids=h, 
                                   attention_mask=h_mask).pooler_output
        else:
            h_e = self.embs(h)
            h_e, *_ = self.hyp_encoder(h_e)
            h_e = torch.max(h_e, 1).values
        
        h_e = self.dropout(h_e)
        return self.classifier(h_e)
    
class HierarchicalReprDialogue(nn.Module):
    def __init__(self, vocab, labels):
        super(HierarchicalReprDialogue, self).__init__()
        self.embs = nn.Embedding(vocab, args.dim1, padding_idx=0)
        
        self.lstm = nn.LSTM(args.dim1, args.dim2, bidirectional=True, batch_first=True)
        self.lstm_hyp = nn.LSTM(args.dim1, args.dim2, bidirectional=True, batch_first=True)
        
        self.lstm_dropout = nn.Dropout(0.3)
        self.turn_drp = nn.Dropout(0.3)
        self.participant_drp = nn.Dropout(0.3)
        
        self.speaker_w = nn.Sequential(nn.Linear(args.dim2, args.dim2), 
                                       nn.ReLU())
        self.listener_w = nn.Sequential(nn.Linear(args.dim2, args.dim2), 
                                        nn.ReLU())
        self.participants_transform = nn.Sequential(nn.Linear(args.dim2, args.dim2), 
                                                    nn.LeakyReLU())
        
        if args.turn_att_module == 'dotp':
            self.turn_att = GeneralAttention(args.dim2, args.dim2)
        elif args.turn_att_module == 'bahd':
            self.turn_att = BahdanauAttention(args.dim2, args.dim2)
        else:
            assert 'Attention Mechanism not available !!!'
        
        self.lstm_transform = nn.Linear(args.dim3, args.dim2)
        
        classifier_dim = args.dim2*3
        # hyp-representation, speaker_representation
        self.classifier = nn.Sequential(nn.Dropout(0.5), 
                                        nn.Linear(classifier_dim, classifier_dim//2), 
                                        nn.LeakyReLU(),
                                        nn.Linear(classifier_dim//2, labels))
        
    def forward(self, p, h, speakers):
        
        participants = torch.zeros(h.size(0), speakers.size(-1), args.dim2, device=DEVICE, requires_grad=True)
        turn_history = torch.zeros(p.size(0), p.size(1), args.dim2, device=DEVICE)
        
        for i in range(p.size(1)):
            curr_turn = p[:,i,:]
            curr_turn = self.embs(curr_turn)
            speaker_idxs = speakers[:,i]
            
            listener_mask = torch.ones(p.size(0), speakers.size(-1), 1, device=DEVICE)
            listener_mask[torch.arange(p.size(0)), speaker_idxs, :] = 0
            speaker_mask = (~listener_mask.bool()).long() 
        
            curr_turn = self.turn_drp(curr_turn)
            turn, *_ = self.lstm(curr_turn)
            hs = self.lstm_dropout(torch.max(turn, 1).values)
            hs = self.lstm_transform(hs)
            
            turn_history[:, i, :] = hs 
            
            #hs = self.lstm_dropout(torch.cat([hs[0,:], hs[1,:]], -1))
            participants_s = self.participant_drp(self.update_state(hs, self.speaker_w, participants, speaker_mask))
            participants_l = self.participant_drp(self.update_state(hs, self.listener_w, participants, listener_mask))
            participants = self.participants_transform(participants_s + participants_l)
            
        h = self.embs(h)
        hyp, (hsh, csh) = self.lstm_hyp(h)
        #hsh = self.lstm_dropout(torch.cat([hsh[0,:], hsh[1,:]], -1))
        hsh = self.lstm_dropout(torch.max(hyp, 1).values)
        hsh = self.lstm_transform(hsh)
        last_speaker = participants[torch.arange(h.size(0)), speakers[:,-1],:]
        
        # key = hsh
        weighted_turns, _ = self.turn_att(hsh, turn_history)
        
        # TODO: attention of participants
        
        # [HYPOTHESIS, SPEAKER_STATE, WEIGHTED_TURNS]       
        final = torch.cat([hsh, last_speaker, weighted_turns], -1)
        #, hsh*last_speaker, abs(hsh-last_speaker)
        
        return self.classifier(final)
    
    def update_state(self, turns: list, transform: list, participants: list, participant_mask: list):
        turns = transform(turns)
        turns = turns.unsqueeze(1).expand(-1,5,-1)
        
        new_speaker_states = self.combine_hiddens(turns, participants, participant_mask)
        return new_speaker_states
        
    def combine_hiddens(self, turns, participants_repr, mask, w = 0.9):
        """
        make w learnable:
        """
        return (w * turns * mask) + participants_repr
    
class BahdanauAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BahdanauAttention, self).__init__()
        self.energy = nn.Parameter(torch.randn(out_dim), requires_grad=True)
        self.Wa = nn.Linear(in_dim, out_dim, bias=False)
        self.Wb = nn.Linear(out_dim, out_dim, bias=False)
        
    def forward(self, k, xs, mask=None):
        ks = k.unsqueeze(1).repeat(1,xs.size(1),1)
        w = torch.tanh(self.Wa(ks) + self.Wb(xs))
        w = w @ self.energy
        
        if mask is not None:
            w.data.masked_fill_(~mask.bool(), float('-inf'))
        
        attn = F.softmax(w, -1)# * mask + EPS
        return torch.einsum('bi,bik->bk', [attn, xs])
    
class GeneralAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GeneralAttention, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        
    def forward(self, k, xs, mask=None):
        ks = k.unsqueeze(1).repeat(1,xs.size(1),1)
        xs = self.W(xs)
        w = torch.matmul(ks, xs.permute(0,2,1))
        
        if mask is not None:
            w.data.masked_fill_(~mask.bool(), float('-inf'))
        
        attn = F.softmax(w, -1)# * mask + EPS
        return torch.einsum('bij,bik->bk', [attn, xs])
            