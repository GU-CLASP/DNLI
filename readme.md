# Dialogue NLI

This repository contains the 2024 DNLI dataset and baseline models published with the following paper:

> Adam Ek, Bill Noble, Stergios Chatzikyriakidis, Robin Cooper, Simon Dobnik, Eleni Gregoromichelaki, Christine Howes, Staffan Larsson, Vladislav Maraev, Gregory Mills, and Gijs Wijnholds. 2024. I hea- umm think that’s what they say: A Dataset of Inferences from Natural Language Dialogues. In _Proceedings of the 28th Workshop on the Semantics and Pragmatics of Dialogue_.

* `data/pretraining` contains the BNC pretraining corpus used to fine-tune BERT. This was created with `create_pretraining_corpus.py`.
* `data/compiled` contains the DNLI dataset provided with different context lengths for context ablation studies. This was created with `create_data_files.py`.
* `baselines` contains code for the the BERT and LSTM baselines.
* `baselines/llm` contains code for the LLama 2 and Zephyr baselines.
