#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试 cmn.txt 数据处理
"""

import os

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

def main():
    # 检查cmn.txt文件是否存在
    if not os.path.exists('data/cmn.txt'):
        print("错误: 找不到 data/cmn.txt 文件")
        return
    
    print("=== 开始处理 cmn.txt 数据 ===")
    
    # 处理Tatoeba数据
    split_tatoeba_data('data/cmn.txt', 'data/train.en', 'data/train.zh')
    
    # 创建验证集
    print("\n=== 创建验证集 ===")
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

    print(f"数据处理完成!")
    print(f"训练集: {len(train_en_final)} 对句子")
    print(f"验证集: {len(valid_en)} 对句子")
    
    # 显示一些样本
    print("\n=== 训练集样本 ===")
    for i in range(min(5, len(train_en_final))):
        print(f"英文: {train_en_final[i].strip()}")
        print(f"中文: {train_zh_final[i].strip()}")
        print("-" * 30)
    
    print("\n=== 验证集样本 ===")
    for i in range(min(3, len(valid_en))):
        print(f"英文: {valid_en[i].strip()}")
        print(f"中文: {valid_zh[i].strip()}")
        print("-" * 30)
    
    print("\n✓ cmn.txt 数据处理成功！现在可以运行完整的训练脚本了。")

if __name__ == "__main__":
    main() 