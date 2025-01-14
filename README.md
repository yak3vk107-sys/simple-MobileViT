# 实验环境
实验使用了ubuntu20.04环境与NVIDIA P106-100 GPU进行训练和测试，python环境如下所示：
```bash
# Create conda environment
conda create -n mobilevit python=3.10
conda activate mobilevit

# Install required packages
pip install -r requirements.txt
```

# 数据集

## 数据集下载
在实验中，我们使用了cifar10和cifar100数据集，可以不在此处下载，训练中代码可以默认检索并下载数据集，运行代码会自动解压数据集，注意将数据集放到data目录下即可。
```bash
# Download cifar10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

# Download cifar100
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

```

# 运行方式

## 训练&测试

```bash
# --resume与后面的内容属于可选内容，如果输入则为断点重连从pth文件中恢复训练
python run.py --resume path/to/best_model.pth
```

## 实验结果

| Model | Dataset | Accuracy | Parameters | FLOPs |
|-------|---------|----------|------------|-------|
| simple-MobileViT | cifar10 | 91.74% | 1.12 M | 7.74 G |
| simple-MobileViT_deeper | cifar10 | 92.51% | 1.12 M | 7.74 G |

## 代码结构
```bash
.
├── data    # 数据集
├── model.py    # 模型整体定义
├── modules
│   ├── __init__.py
│   ├── mixup_data.py   # 数据增强函数
│   ├── mnetv2.py       # mobilenetv2块
│   ├── mobilevit_block_test.py   # mobilvevit块
│   └── __pycache__
│      
│       
│       
│       
├── output              # 结果存放在这里
│   └── image_enhance
│       └── mixup_cifar100_baseline
│           ├── best_model.pth
│           └── checkpoint.pth
├── __pycache__
├── README.md
├── requirements.txt    # python环境
└── run.py              # 训练&测试代码
```
