#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更好的训练脚本
使用 cmn.txt (Tatoeba) 数据训练 Transformer 模型
包含更好的超参数设置和训练策略
"""

import sys
import os
import torch
import numpy as np
import pickle
import logging
import time
import math
import re
import unicodedata
from collections import Counter

# 数据处理相关
import jieba
import nltk
from tqdm import tqdm

# PyTorch相关
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# 添加当前目录到Python路径
sys.path.append('.')

# 导入项目模块
from config import *
from utils import *
from data_gen_fixed import AiChallenger2017Dataset, pad_collate
from transformer.transformer import Transformer
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"设备: {device}")

def split_tatoeba_data(input_file, output_en, output_zh):
    """
    将Tatoeba格式的数据文件分离成英文和中文两个文件
    """
    print(f"正在处理: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    en_count = 0
    zh_count = 0
    
    with open(output_en, 'w', encoding='utf-8') as f_en:
        with open(output_zh, 'w', encoding='utf-8') as f_zh:
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    en_text = parts[0].strip()
                    zh_text = parts[1].strip()
                    # 过滤掉许可证信息等
                    if not en_text.startswith('CC-BY') and not zh_text.startswith('CC-BY'):
                        f_en.write(en_text + '\n')
                        f_zh.write(zh_text + '\n')
                        en_count += 1
                        zh_count += 1
    
    print(f"分离完成! 英文句子: {en_count}, 中文句子: {zh_count}")

def process(file, lang='zh'):
    print('processing {}...'.format(file))
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()

    word_freq = Counter()
    lengths = []

    for line in tqdm(data):
        sentence = line.strip()
        if lang == 'en':
            sentence_en = sentence.lower()
            tokens = [normalizeString(s) for s in nltk.word_tokenize(sentence_en)]
            word_freq.update(list(tokens))
            vocab_size = n_src_vocab
        else:
            seg_list = jieba.cut(sentence.strip())
            tokens = list(seg_list)
            word_freq.update(list(tokens))
            vocab_size = n_tgt_vocab

        lengths.append(len(tokens))

    words = word_freq.most_common(vocab_size - 4)
    word_map = {k[0]: v + 4 for v, k in enumerate(words)}
    word_map['<pad>'] = 0
    word_map['<sos>'] = 1
    word_map['<eos>'] = 2
    word_map['<unk>'] = 3
    print(len(word_map))
    print(words[:100])

    word2idx = word_map
    idx2char = {v: k for k, v in word2idx.items()}

    return word2idx, idx2char

def get_data(in_file, out_file):
    print('getting data {}->{}...'.format(in_file, out_file))
    with open(in_file, 'r', encoding='utf-8') as file:
        in_lines = file.readlines()
    with open(out_file, 'r', encoding='utf-8') as file:
        out_lines = file.readlines()

    samples = []

    for i in tqdm(range(len(in_lines))):
        sentence_en = in_lines[i].strip().lower()
        tokens = [normalizeString(s.strip()) for s in nltk.word_tokenize(sentence_en)]
        in_data = encode_text(src_char2idx, tokens)

        sentence_zh = out_lines[i].strip()
        tokens = jieba.cut(sentence_zh.strip())
        out_data = [sos_id] + encode_text(tgt_char2idx, tokens) + [eos_id]

        if len(in_data) < maxlen_in and len(out_data) < maxlen_out and unk_id not in in_data and unk_id not in out_data:
            samples.append({'in': in_data, 'out': out_data})
    return samples

def main():
    # 步骤1: 检查cmn.txt文件是否存在
    if not os.path.exists('data/cmn.txt'):
        print("错误: 找不到 data/cmn.txt 文件")
        return
    
    # 步骤2: 处理Tatoeba数据
    print("=== 步骤1: 处理Tatoeba数据 ===")
    split_tatoeba_data('data/cmn.txt', 'data/train.en', 'data/train.zh')
    
    # 步骤3: 创建验证集
    print("\n=== 步骤2: 创建验证集 ===")
    with open('data/train.en', 'r', encoding='utf-8') as f:
        train_en_lines = f.readlines()
    with open('data/train.zh', 'r', encoding='utf-8') as f:
        train_zh_lines = f.readlines()

    # 取最后10%作为验证集
    split_point = int(len(train_en_lines) * 0.9)
    train_en_final = train_en_lines[:split_point]
    train_zh_final = train_zh_lines[:split_point]
    valid_en = train_en_lines[split_point:]
    valid_zh = train_zh_lines[split_point:]

    # 重写训练文件
    with open('data/train.en', 'w', encoding='utf-8') as f:
        f.writelines(train_en_final)
    with open('data/train.zh', 'w', encoding='utf-8') as f:
        f.writelines(train_zh_final)

    # 创建验证文件
    with open('data/valid.en', 'w', encoding='utf-8') as f:
        f.writelines(valid_en)
    with open('data/valid.zh', 'w', encoding='utf-8') as f:
        f.writelines(valid_zh)

    print(f"数据处理完成! 训练集: {len(train_en_final)} 对句子, 验证集: {len(valid_en)} 对句子")
    
    # 步骤4: 数据预处理
    print("\n=== 步骤3: 数据预处理 ===")
    global src_char2idx, tgt_char2idx
    src_char2idx, src_idx2char = process(train_translation_en_filename, lang='en')
    tgt_char2idx, tgt_idx2char = process(train_translation_zh_filename, lang='zh')

    print(f"英文词汇表大小: {len(src_char2idx)}")
    print(f"中文词汇表大小: {len(tgt_char2idx)}")

    data = {
        'dict': {
            'src_char2idx': src_char2idx,
            'src_idx2char': src_idx2char,
            'tgt_char2idx': tgt_char2idx,
            'tgt_idx2char': tgt_idx2char
        }
    }
    with open(vocab_file, 'wb') as file:
        pickle.dump(data, file)

    train = get_data(train_translation_en_filename, train_translation_zh_filename)
    valid = get_data(valid_translation_en_filename, valid_translation_zh_filename)

    data = {
        'train': train,
        'valid': valid
    }

    print(f'训练样本数: {len(train)}')
    print(f'验证样本数: {len(valid)}')

    with open(data_file, 'wb') as file:
        pickle.dump(data, file)

    print("数据预处理完成!")
    
    # 步骤5: 开始训练
    print("\n=== 步骤4: 开始训练 ===")
    
    # 设置更好的训练参数
    class Args:
        def __init__(self):
            # 减小模型规模，防止过拟合
            self.n_layers_enc = 4  # 从6减少到4
            self.n_head = 8
            self.d_k = 64
            self.d_v = 64
            self.d_model = 256  # 从512减少到256
            self.d_inner = 1024  # 从2048减少到1024
            self.dropout = 0.3  # 增加dropout
            self.pe_maxlen = 5000
            self.d_word_vec = 256  # 从512减少到256
            self.n_layers_dec = 4  # 从6减少到4
            self.tgt_emb_prj_weight_sharing = 1
            self.label_smoothing = 0.1
            self.epochs = 50  # 增加训练轮数
            self.shuffle = 1
            self.batch_size = 8  # 进一步减小batch size
            self.batch_frames = 0
            self.maxlen_in = 50
            self.maxlen_out = 25
            self.num_workers = 2
            self.k = 0.2
            self.warmup_steps = 8000  # 增加warmup步数
            self.checkpoint = None
            self.patience = 8  # 增加早停耐心值
            self.min_lr = 1e-6  # 最小学习率

    args = Args()

    print(f"训练参数: epochs={args.epochs}, batch_size={args.batch_size}, dropout={args.dropout}")
    print(f"模型参数: d_model={args.d_model}, d_inner={args.d_inner}, layers={args.n_layers_enc}")
    
    # 开始训练
    train_net_better(args)
    
    print("训练完成!")

def train_net_better(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter('logs')
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        # model
        encoder = Encoder(n_src_vocab, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(sos_id, eos_id, n_tgt_vocab,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)
        model = Transformer(encoder, decoder)

        # optimizer with weight decay and lower learning rate
        optimizer = TransformerOptimizer(
            torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, lr=1e-4))

    else:
        checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to GPU, if available
    model = model.to(device)

    # Custom dataloaders
    train_dataset = AiChallenger2017Dataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               shuffle=True, num_workers=args.num_workers)
    valid_dataset = AiChallenger2017Dataset('valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               shuffle=False, num_workers=args.num_workers)

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        train_loss = train_better(train_loader=train_loader,
                                 model=model,
                                 optimizer=optimizer,
                                 epoch=epoch,
                                 logger=logger,
                                 writer=writer)

        writer.add_scalar('epoch/train_loss', train_loss, epoch)
        writer.add_scalar('epoch/learning_rate', optimizer.lr, epoch)

        print('\nLearning rate: {}'.format(optimizer.lr))
        print('Step num: {}\n'.format(optimizer.step_num))

        # One epoch's validation
        valid_loss = valid_better(valid_loader=valid_loader,
                                 model=model,
                                 logger=logger)
        writer.add_scalar('epoch/valid_loss', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            
            # 学习率衰减
            if epochs_since_improvement >= 3:
                for param_group in optimizer.optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.8, args.min_lr)
                print(f"学习率衰减到: {optimizer.optimizer.param_groups[0]['lr']}")
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)
        
        # Early stopping
        if epochs_since_improvement >= args.patience:
            print(f"\n早停触发! {args.patience} 轮没有改善，停止训练")
            break

def train_better(train_loader, model, optimizer, epoch, logger, writer):
    model.train()

    losses = AverageMeter()
    times = AverageMeter()

    start = time.time()

    # Batches
    for i, (data) in enumerate(train_loader):
        # Move to GPU, if available
        padded_input, padded_target, input_lengths = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        input_lengths = input_lengths.to(device)

        # Forward prop.
        pred, gold = model(padded_input, input_lengths, padded_target)
        loss, n_correct = cal_performance(pred, gold, smoothing=0.1)
        try:
            assert (not math.isnan(loss.item()))
        except AssertionError:
            print('n_correct: ' + str(n_correct))
            print('data: ' + str(n_correct))
            continue

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer.optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        elapsed = time.time() - start
        start = time.time()

        losses.update(loss.item())
        times.update(elapsed)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Batch time {time.val:.5f} ({time.avg:.5f})\t'
                        'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, i, len(train_loader), time=times,
                                                                      loss=losses))
            writer.add_scalar('step_num/train_loss', losses.avg, optimizer.step_num)
            writer.add_scalar('step_num/learning_rate', optimizer.lr, optimizer.step_num)

    return losses.avg

def valid_better(valid_loader, model, logger):
    model.eval()

    losses = AverageMeter()

    # Batches
    for data in valid_loader:
        # Move to GPU, if available
        padded_input, padded_target, input_lengths = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        input_lengths = input_lengths.to(device)

        with torch.no_grad():
            # Forward prop.
            pred, gold = model(padded_input, input_lengths, padded_target)
            loss, n_correct = cal_performance(pred, gold, smoothing=0.1)
            try:
                assert (not math.isnan(loss.item()))
            except AssertionError:
                print('n_correct: ' + str(n_correct))
                print('data: ' + str(n_correct))
                continue

        # Keep track of metrics
        losses.update(loss.item())

    # Print status
    logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))

    return losses.avg

def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best):
    """
    保存模型检查点
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'model': model,
             'optimizer': optimizer,
             'best_loss': best_loss, }
    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')

if __name__ == "__main__":
    main() 