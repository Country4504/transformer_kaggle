# Kaggle环境配置说明

## 路径修改总结

为了适应Kaggle环境，我们对以下文件进行了路径修改：

### 1. 主要配置文件修改

#### `config.py` 和 `config_kaggle.py`
- 添加了 `KAGGLE_BASE_PATH = '/kaggle/working/Transformer'`
- 修改了数据文件路径：
  - `data_file` 和 `vocab_file` 使用绝对路径
  - 训练和验证数据文件路径根据实际位置调整

#### `utils.py`
- 添加了Kaggle路径配置
- 修改了checkpoint保存路径

#### `train.py`
- 添加了Kaggle路径配置
- 修改了TensorBoard日志路径

#### `export.py`
- 修改了checkpoint加载路径

### 2. 数据文件路径配置选项

根据您的数据实际位置，可以选择以下配置：

#### 选项1: 数据在Kaggle的input目录中
```python
train_translation_en_filename = '/kaggle/input/ai-challenger-translation/train.en'
train_translation_zh_filename = '/kaggle/input/ai-challenger-translation/train.zh'
valid_translation_en_filename = '/kaggle/input/ai-challenger-translation/valid.en'
valid_translation_zh_filename = '/kaggle/input/ai-challenger-translation/valid.zh'
```

#### 选项2: 数据在working目录中
```python
train_translation_en_filename = '/kaggle/working/Transformer/data/ai_challenger_translation_train_20170904/translation_train_data_20170904/train.en'
train_translation_zh_filename = '/kaggle/working/Transformer/data/ai_challenger_translation_train_20170904/translation_train_data_20170904/train.zh'
valid_translation_en_filename = '/kaggle/working/Transformer/data/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.en'
valid_translation_zh_filename = '/kaggle/working/Transformer/data/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.zh'
```

### 3. 在Jupyter Notebook中的使用

#### 导入配置
```python
# 使用Kaggle专用配置
from config_kaggle import *

# 或者使用修改后的原配置
from config import *
```

#### 路径检查
```python
# 检查数据文件是否存在
import os
data_files = [
    train_translation_en_filename,
    train_translation_zh_filename,
    valid_translation_en_filename,
    valid_translation_zh_filename
]

for file_path in data_files:
    if os.path.exists(file_path):
        print(f"✓ {file_path}")
    else:
        print(f"✗ {file_path} (文件不存在)")
```

### 4. 文件结构

在Kaggle环境中，建议的文件结构：

```
/kaggle/working/Transformer/
├── config.py              # 原配置文件
├── config_kaggle.py       # Kaggle专用配置
├── utils.py               # 工具函数
├── data_gen.py            # 数据生成器
├── pre_process.py         # 数据预处理
├── train.py               # 训练脚本
├── demo.py                # 演示脚本
├── transformer/           # Transformer模型实现
│   ├── encoder.py
│   ├── decoder.py
│   ├── attention.py
│   ├── loss.py
│   ├── optimizer.py
│   └── transformer.py
├── data.pkl               # 预处理后的数据
├── vocab.pkl              # 词汇表
├── checkpoint.tar         # 训练检查点
├── BEST_checkpoint.tar    # 最佳模型
└── logs/                  # TensorBoard日志
```

### 5. 使用步骤

1. **上传项目文件**：将所有项目文件上传到Kaggle的working目录
2. **配置数据路径**：根据实际数据位置修改`config_kaggle.py`中的路径
3. **安装依赖**：在notebook中安装必要的Python包
4. **运行预处理**：执行数据预处理脚本
5. **开始训练**：运行模型训练
6. **保存结果**：将重要文件保存到Kaggle输出

### 6. 注意事项

- 确保数据文件路径正确
- 检查GPU可用性
- 注意Kaggle的内存和运行时间限制
- 定期保存训练检查点
- 使用相对路径时注意当前工作目录 