# ============================================================================
# 单元格 1: 环境设置和依赖安装
# ============================================================================

# 安装必要的依赖包
# !pip install jieba nltk tqdm

# 下载nltk数据
import nltk
nltk.download('punkt')

# ============================================================================
# 单元格 2: 导入必要的库
# ============================================================================

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
sys.path.append('/kaggle/working/Transformer')

# 导入项目模块
from config import *
from utils import *
from data_gen import AiChallenger2017Dataset, pad_collate
from transformer.transformer import Transformer
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"设备: {device}")

# ============================================================================
# 单元格 3: 数据路径检查
# ============================================================================

# 检查数据文件是否存在
data_files = [
    train_translation_en_filename,
    train_translation_zh_filename,
    valid_translation_en_filename,
    valid_translation_zh_filename
]

print("数据文件检查:")
for file_path in data_files:
    if os.path.exists(file_path):
        print(f"✓ {file_path}")
    else:
        print(f"✗ {file_path} (文件不存在)")

# ============================================================================
# 单元格 4: 数据分离函数 (处理Tatoeba格式数据)
# ============================================================================

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

# 处理Tatoeba数据 - 使用cmn.txt文件
print("开始处理Tatoeba数据...")
split_tatoeba_data('data/cmn.txt', 'data/train.en', 'data/train.zh')

# 创建验证集（从训练数据中取一部分）
print("创建验证集...")
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

# ============================================================================
# 单元格 5: 数据预处理 (pre_process.py)
# ============================================================================

def build_vocab(token, word2idx, idx2char):
    if token not in word2idx:
        next_index = len(word2idx)
        word2idx[token] = next_index
        idx2char[next_index] = token

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

# 开始数据预处理
print("开始数据预处理...")
src_char2idx, src_idx2char = process(train_translation_en_filename, lang='en')
tgt_char2idx, tgt_idx2char = process(train_translation_zh_filename, lang='zh')

print(len(src_char2idx))
print(len(tgt_char2idx))

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

print('num_train: ' + str(len(train)))
print('num_valid: ' + str(len(valid)))

with open(data_file, 'wb') as file:
    pickle.dump(data, file)

print("数据预处理完成!")

# ============================================================================
# 单元格 6: 检查预处理结果
# ============================================================================

# 检查生成的文件
print("检查预处理结果：")
if os.path.exists('vocab.pkl'):
    print("✓ vocab.pkl 已生成")
    with open('vocab.pkl', 'rb') as f:
        vocab_data = pickle.load(f)
    print(f"词汇表大小: 英文 {len(vocab_data['dict']['src_char2idx'])} 个词")
    print(f"词汇表大小: 中文 {len(vocab_data['dict']['tgt_char2idx'])} 个词")

if os.path.exists('data.pkl'):
    print("✓ data.pkl 已生成")
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"训练样本数: {len(data['train'])}")
    print(f"验证样本数: {len(data['valid'])}")

# ============================================================================
# 单元格 7: 模型训练 (train.py)
# ============================================================================

def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter(os.path.join(KAGGLE_BASE_PATH, 'logs'))
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

        # optimizer
        optimizer = TransformerOptimizer(
            torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09))

    else:
        checkpoint = torch.load(checkpoint)
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
        train_loss = train(train_loader=train_loader,
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
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           logger=logger)
        writer.add_scalar('epoch/valid_loss', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)

def train(train_loader, model, optimizer, epoch, logger, writer):
    model.train()  # train mode (dropout and batchnorm is used)

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
        loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)
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

def valid(valid_loader, model, logger):
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
            loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)
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

# ============================================================================
# 单元格 8: 开始训练
# ============================================================================

# 设置训练参数
class Args:
    def __init__(self):
        self.n_layers_enc = 6
        self.n_head = 8
        self.d_k = 64
        self.d_v = 64
        self.d_model = 512
        self.d_inner = 2048
        self.dropout = 0.1
        self.pe_maxlen = 5000
        self.d_word_vec = 512
        self.n_layers_dec = 6
        self.tgt_emb_prj_weight_sharing = 1
        self.label_smoothing = 0.1
        self.epochs = 10  # 先用较少的epochs测试
        self.shuffle = 1
        self.batch_size = 64
        self.batch_frames = 0
        self.maxlen_in = 50
        self.maxlen_out = 25
        self.num_workers = 4
        self.k = 0.2
        self.warmup_steps = 4000
        self.checkpoint = None

args = Args()

print("开始模型训练...")
print(f"训练参数: epochs={args.epochs}, batch_size={args.batch_size}")

# 开始训练
train_net(args)

print("训练完成!")

# ============================================================================
# 单元格 9: 模型测试 (demo.py)
# ============================================================================

def translate_sentence(sentence, model, src_char2idx, tgt_idx2char, device):
    """
    翻译单个句子
    """
    model.eval()
    
    # 预处理输入句子
    sentence = sentence.lower()
    tokens = [normalizeString(s.strip()) for s in nltk.word_tokenize(sentence)]
    in_data = encode_text(src_char2idx, tokens)
    
    # 转换为tensor
    src = torch.LongTensor(in_data).unsqueeze(0).to(device)
    src_length = torch.LongTensor([len(in_data)]).to(device)
    
    with torch.no_grad():
        # 翻译
        pred = model(src, src_length, None, teacher_forcing_ratio=0.0)
        pred = pred.max(1)[1]
        
        # 转换为文本
        pred_text = []
        for idx in pred[0]:
            if idx.item() == eos_id:
                break
            if idx.item() != sos_id:
                pred_text.append(tgt_idx2char[idx.item()])
    
    return ''.join(pred_text)

# 加载训练好的模型
checkpoint_path = os.path.join(KAGGLE_BASE_PATH, 'BEST_checkpoint.tar')
if os.path.exists(checkpoint_path):
    print("加载训练好的模型...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = checkpoint['model']
    
    # 加载词汇表
    with open(vocab_file, 'rb') as f:
        vocab_data = pickle.load(f)
    src_char2idx = vocab_data['dict']['src_char2idx']
    tgt_idx2char = vocab_data['dict']['tgt_idx2char']
    
    # 测试翻译
    test_sentences = [
        "Hello world",
        "How are you",
        "I love you",
        "Thank you very much"
    ]
    
    print("\n翻译测试:")
    for sentence in test_sentences:
        translation = translate_sentence(sentence, model, src_char2idx, tgt_idx2char, device)
        print(f"英文: {sentence}")
        print(f"中文: {translation}")
        print("-" * 30)
else:
    print("未找到训练好的模型，请先完成训练")

# ============================================================================
# 单元格 10: 保存结果
# ============================================================================

# 检查生成的文件
print("生成的文件:")
import shutil

# 保存重要文件到Kaggle输出
if os.path.exists(os.path.join(KAGGLE_BASE_PATH, 'BEST_checkpoint.tar')):
    shutil.copy(os.path.join(KAGGLE_BASE_PATH, 'BEST_checkpoint.tar'), '/kaggle/working/')
    print("最佳模型已保存到 /kaggle/working/BEST_checkpoint.tar")

if os.path.exists(vocab_file):
    shutil.copy(vocab_file, '/kaggle/working/')
    print("词汇表已保存到 /kaggle/working/vocab.pkl")

if os.path.exists(data_file):
    shutil.copy(data_file, '/kaggle/working/')
    print("处理后的数据已保存到 /kaggle/working/data.pkl")

print("\n项目运行完成!") 