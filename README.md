# 面部表情识别 (Facial Expression Recognition)

基于PyTorch的面部表情识别系统，使用FER2013数据集训练简单CNN和ResNet模型来识别7种基本表情。

## 项目概述

这个项目实现了一个面部表情识别系统，能够将面部图像分类为以下7种表情之一:
- 生气 (Angry)
- 厌恶 (Disgust)
- 恐惧 (Fear)
- 高兴 (Happy)
- 中性 (Neutral)
- 悲伤 (Sad)
- 惊讶 (Surprise)

## 数据集

本项目使用FER2013数据集，该数据集包含48x48像素的灰度人脸图像，每张图像被标注为7种基本情绪之一。数据集已被分为训练集和测试集，目录结构如下:

```
FER2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

## 项目结构

```
├── FER2013/               # 数据集目录
├── models/                # 模型定义
│   ├── __init__.py
│   └── SimpleCNN.py       # 简单CNN模型实现
├── checkpoints/           # 保存训练好的模型
│   └── simplecnn_fer2013.pth
├── logs/                  # 训练日志和可视化
│   └── fer2013_simplecnn/
│       ├── accuracy_plot.png
│       └── ...
├── pretrained/            # 预训练模型权重
│   ├── resnet18-5c106cde.pth
│   └── resnet50-0676ba61.pth
├── scripts/               # 训练脚本
│   ├── train_resnet18_pretrained.sh
│   └── train_simple_cnn.sh
├── fer2013.ipynb          # Jupyter notebook
└── train.py               # 训练脚本
```

## 模型

本项目实现了两种模型:

1. **SimpleCNN** - 一个简单的卷积神经网络，专为FER2013数据集设计
2. **ResNet** - 使用迁移学习的ResNet18/ResNet50模型

## 安装与依赖

```bash
# 克隆仓库
git clone https://github.com/SuibeAI/FacialExpressionRecognition.git
cd FacialExpressionRecognition


# 安装依赖
pip install torch torchvision matplotlib sklearn jupyter tensorboard
```

## 解压数据集
```
cd FER2013
unzip archives.zip
cd ..  
```

## 下载模型
```
cd pretrained
bash download.sh
cd ..
```


## 使用方法

### 训练模型

使用SimpleCNN训练:

```bash
python train.py --model_type simplecnn --image_size 48 --num_input_channels 1
```

使用ResNet18训练:

```bash
python train.py --model_type resnet18 --image_size 224 --num_input_channels 3 --pretrained_path ./pretrained/resnet18-5c106cde.pth
```

也可以使用提供的脚本:

```bash
bash scripts/train_simple_cnn.sh
# 或
bash scripts/train_resnet18_pretrained.sh
```

### 参数说明

- `--model_type`: 选择模型类型 (`simplecnn` 或 `resnet18`)
- `--image_size`: 输入图像大小 (SimpleCNN推荐48，ResNet推荐224)
- `--num_input_channels`: 输入通道数 (灰度=1，彩色=3)
- `--pretrained_path`: 预训练模型路径 (可选)
- `--use_tiny_dataset`: 使用小数据集进行快速验证 (可选)

### 查看训练结果

训练过程中会生成以下可视化结果:

1. 训练和验证准确率曲线
2. 训练和验证损失曲线
3. 混淆矩阵

所有可视化结果保存在`logs/fer2013_<model_type>/`目录下。

训练日志可通过TensorBoard查看:

```bash
tensorboard --logdir=logs/
```

## 项目亮点

1. **早停机制**: 当验证损失在连续几个epoch没有改善时，训练会提前结束
2. **模型检查点**: 保存验证损失最低的模型
3. **性能可视化**: 使用TensorBoard记录训练过程
4. **灵活配置**: 支持多种模型架构和配置选项


