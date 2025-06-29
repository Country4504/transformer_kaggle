import logging
import os

import torch

# Kaggle环境路径配置
KAGGLE_BASE_PATH = '/kaggle/working/Transformer'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
d_model = 512
epochs = 10000
embedding_size = 300
hidden_size = 1024
data_file = os.path.join(KAGGLE_BASE_PATH, 'data.pkl')
vocab_file = os.path.join(KAGGLE_BASE_PATH, 'vocab.pkl')
n_src_vocab = 15000
n_tgt_vocab = 15000  # target
maxlen_in = 100
maxlen_out = 50
# Training parameters
grad_clip = 1.0  # clip gradients at an absolute value of
print_freq = 50  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
IGNORE_ID = -1
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
num_train = 4669414
num_valid = 3870

# 数据文件路径配置

#如果数据在working目录中
train_translation_en_filename = os.path.join(KAGGLE_BASE_PATH, 'data/train.en')
train_translation_zh_filename = os.path.join(KAGGLE_BASE_PATH, 'data/train.zh')
valid_translation_en_filename = os.path.join(KAGGLE_BASE_PATH, 'data/valid.en')
valid_translation_zh_filename = os.path.join(KAGGLE_BASE_PATH, 'data/valid.zh')

# 选项3: 如果数据在其他位置，请修改以下路径
# train_translation_en_filename = '/path/to/your/train.en'
# train_translation_zh_filename = '/path/to/your/train.zh'
# valid_translation_en_filename = '/path/to/your/valid.en'
# valid_translation_zh_filename = '/path/to/your/valid.zh'


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger() 