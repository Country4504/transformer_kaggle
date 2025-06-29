import time
import os

import torch

from transformer.transformer import Transformer
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from config import n_src_vocab, n_tgt_vocab, sos_id, eos_id, d_model, device

# Kaggle环境路径配置
KAGGLE_BASE_PATH = '/kaggle/working/Transformer'
checkpoint = os.path.join(KAGGLE_BASE_PATH, 'BEST_checkpoint.tar')

if __name__ == '__main__':
    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    model = checkpoint['model']
    print(type(model))

    filename = 'transformer.pt'
    print('saving {}...'.format(filename))
    start = time.time()
    torch.save(model.state_dict(), filename)
    print('elapsed {} sec'.format(time.time() - start))

    print('loading {}...'.format(filename))
    start = time.time()
    model = Transformer()
    model.load_state_dict(torch.load(filename))
    print('elapsed {} sec'.format(time.time() - start))
