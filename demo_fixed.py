#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版本的翻译演示脚本
使用训练好的模型进行英文到中文翻译
"""

import pickle
import random
import time
import sys
import os

import numpy as np
import torch

# 添加当前目录到Python路径
sys.path.append('.')

from config import device, logger, data_file, vocab_file, n_src_vocab, n_tgt_vocab, sos_id, eos_id
from transformer.transformer import Transformer
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from utils import normalizeString, encode_text, sequence_to_text

def translate_sentence(sentence, model, src_char2idx, tgt_idx2char, device):
    """
    翻译单个句子
    """
    model.eval()
    
    # 预处理输入句子
    sentence = sentence.lower()
    tokens = [normalizeString(s.strip()) for s in sentence.split()]
    in_data = encode_text(src_char2idx, tokens)
    
    # 转换为tensor
    src = torch.LongTensor(in_data).unsqueeze(0).to(device)
    src_length = torch.LongTensor([len(in_data)]).to(device)
    
    with torch.no_grad():
        # 使用recognize方法进行翻译
        nbest_hyps = model.recognize(src.squeeze(0), src_length, tgt_idx2char)
        
        # 获取最佳翻译结果
        if nbest_hyps and len(nbest_hyps) > 0:
            best_hyp = nbest_hyps[0]
            pred_text = []
            for idx in best_hyp['yseq']:
                if idx == eos_id:
                    break
                if idx != sos_id:
                    pred_text.append(tgt_idx2char[idx])
            return ''.join(pred_text)
        else:
            return "翻译失败"

def main():
    # 检查检查点文件是否存在
    checkpoint_file = 'BEST_checkpoint.tar'
    if not os.path.exists(checkpoint_file):
        print(f"错误: 找不到检查点文件 {checkpoint_file}")
        print("请先完成模型训练")
        return
    
    print(f'加载检查点文件: {checkpoint_file}')
    start = time.time()
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    model = checkpoint['model']
    print(f'加载完成，耗时: {time.time() - start:.2f} 秒')
    
    model = model.to(device)
    model.eval()
    
    # 加载词汇表
    print('加载词汇表...')
    with open(vocab_file, 'rb') as f:
        vocab_data = pickle.load(f)
    src_char2idx = vocab_data['dict']['src_char2idx']
    tgt_idx2char = vocab_data['dict']['tgt_idx2char']
    
    print(f'英文词汇表大小: {len(src_char2idx)}')
    print(f'中文词汇表大小: {len(tgt_idx2char)}')
    
    # 测试翻译
    test_sentences = [
        "Hello world",
        "How are you",
        "I love you",
        "Thank you very much",
        "Good morning",
        "What is your name",
        "I am happy",
        "The weather is nice today",
        "Can you help me",
        "See you tomorrow"
    ]
    
    print("\n" + "="*50)
    print("翻译测试结果:")
    print("="*50)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. 英文: {sentence}")
        try:
            translation = translate_sentence(sentence, model, src_char2idx, tgt_idx2char, device)
            print(f"   中文: {translation}")
        except Exception as e:
            print(f"   翻译失败: {e}")
        print("-" * 30)
    
    # 交互式翻译
    print("\n" + "="*50)
    print("交互式翻译模式 (输入 'quit' 退出):")
    print("="*50)
    
    while True:
        try:
            user_input = input("\n请输入英文句子: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue
                
            translation = translate_sentence(user_input, model, src_char2idx, tgt_idx2char, device)
            print(f"翻译结果: {translation}")
            
        except KeyboardInterrupt:
            print("\n\n退出翻译模式")
            break
        except Exception as e:
            print(f"翻译出错: {e}")

if __name__ == "__main__":
    main() 