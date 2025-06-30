#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 cmn.txt 数据处理
"""

import os
import sys

def test_cmn_data():
    """测试cmn.txt数据的处理"""
    
    # 检查文件是否存在
    cmn_file = 'data/cmn.txt'
    if not os.path.exists(cmn_file):
        print(f"错误: 找不到文件 {cmn_file}")
        return False
    
    print(f"✓ 找到文件: {cmn_file}")
    
    # 读取前几行查看格式
    with open(cmn_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"文件总行数: {len(lines)}")
    print("\n前5行内容:")
    for i, line in enumerate(lines[:5]):
        print(f"第{i+1}行: {line.strip()}")
    
    # 分析数据格式
    valid_pairs = 0
    invalid_pairs = 0
    
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            en_text = parts[0].strip()
            zh_text = parts[1].strip()
            # 过滤掉许可证信息等
            if not en_text.startswith('CC-BY') and not zh_text.startswith('CC-BY'):
                valid_pairs += 1
            else:
                invalid_pairs += 1
        else:
            invalid_pairs += 1
    
    print(f"\n数据统计:")
    print(f"有效翻译对: {valid_pairs}")
    print(f"无效行: {invalid_pairs}")
    print(f"有效数据比例: {valid_pairs/(valid_pairs+invalid_pairs)*100:.2f}%")
    
    return True

def test_data_processing():
    """测试数据处理流程"""
    
    print("\n=== 测试数据处理流程 ===")
    
    # 模拟数据处理
    from run_with_cmn import split_tatoeba_data
    
    # 创建临时文件
    temp_en = 'temp_train.en'
    temp_zh = 'temp_train.zh'
    
    try:
        # 处理数据
        split_tatoeba_data('data/cmn.txt', temp_en, temp_zh)
        
        # 检查生成的文件
        if os.path.exists(temp_en) and os.path.exists(temp_zh):
            with open(temp_en, 'r', encoding='utf-8') as f:
                en_lines = f.readlines()
            with open(temp_zh, 'r', encoding='utf-8') as f:
                zh_lines = f.readlines()
            
            print(f"✓ 成功生成训练文件")
            print(f"英文句子数: {len(en_lines)}")
            print(f"中文句子数: {len(zh_lines)}")
            
            # 显示几个样本
            print("\n前3个翻译样本:")
            for i in range(min(3, len(en_lines))):
                print(f"英文: {en_lines[i].strip()}")
                print(f"中文: {zh_lines[i].strip()}")
                print("-" * 30)
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_en):
            os.remove(temp_en)
        if os.path.exists(temp_zh):
            os.remove(temp_zh)

if __name__ == "__main__":
    print("开始测试 cmn.txt 数据...")
    
    if test_cmn_data():
        test_data_processing()
        print("\n✓ 测试完成! cmn.txt 数据可以正常使用")
    else:
        print("\n✗ 测试失败!") 