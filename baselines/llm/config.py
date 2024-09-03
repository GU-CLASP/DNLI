import socket
import os
import torch

def bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

def awq_config():
    from transformers import AwqConfig
    return AwqConfig()

def set_environ(config):
    if 'environ' in config:
        for var in config['environ']:
            os.environ[var] = config['environ'][var]

def set_quantization_config(config):
    if 'quantization_config' in config['model-config']:
        config['model-config']['quantization_config'] = config['model-config']['quantization_config']() 
        

configs = {

    'hostname': {
        'default-model': 'TheBloke/Llama-2-7B-AWQ',
        'model-config': {
            'echo': False,
            'device_map': 'auto',
        },
    },

}


# hostname = socket.gethostname()
hostname = 'hostname'
config = configs[hostname]
set_environ(config)
set_quantization_config(config)
