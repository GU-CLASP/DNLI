from argparse import ArgumentParser

a_parser = ArgumentParser()

## model architecture
a_parser.add_argument('--dim1', type=int, default=128)
a_parser.add_argument('--dim2', type=int, default=256)
a_parser.add_argument('--dim3', type=int, default=512)

a_parser.add_argument('--data_type', 
                      type=str, 
                      choices=['flat', 'hierarchical'], 
                      default='hierarchical')
a_parser.add_argument('--model_configuration', 
                      type=str, 
                      choices=['bert_flat', 'lstm_flat', 
                               'bert_hierarchical', 'lstm_hierarchical', 
                               'hyp_only_bert', 'hyp_only_lstm', 
                               'hierarchical_transformer', 'flat_transformer'], 
                      default='bert_hierarchical')
                               
a_parser.add_argument('--turn_pooling', 
                      type=str, 
                      choices=['att', 'max', 'self_att'], 
                      default='att')

a_parser.add_argument('--token_pooling', 
                      type=str, 
                      choices=['att', 'max', 'self_att'], 
                      default='self_att')

a_parser.add_argument('--remove_dialogue_phenomenon', 
                      type=str, 
                      choices=['all', 'backchannel', 'disfluencies'], 
                      default='max')

## learning rate
a_parser.add_argument('--lr',           type=float, default=0.0001)
a_parser.add_argument('--bert_lr',      type=float, default=0.00001) #0.00001 
a_parser.add_argument('--min_lr',       type=float, default=0.0000000001) # .000000001
a_parser.add_argument('--weight_decay', type=float, default=0.0) # prev: 0.01

## other hyperparams
a_parser.add_argument('--n_epochs',      type=int,   default=10)
a_parser.add_argument('--batch_size',    type=int,   default=64)
a_parser.add_argument('--clipping',      type=float, default=5.0)
a_parser.add_argument('--train_verbose', type=bool,  default=True)

## etc
a_parser.add_argument('--seed',   type=int, default=333)
a_parser.add_argument('--device', type=str, default='cuda:0')

args = a_parser.parse_args()



