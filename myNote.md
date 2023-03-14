# Contents:

- [过拟合](#ovft)
- [构建网络](#net)
- [损失函数](#loss)
- [训练](#train)

## 过拟合<div id="ovft"></div>

### 如何判断：

- 训练集上表现好，测试集表现差。
- 模型泛化能力弱。
- 参数过多。

### 解决办法：

- 增加数据量和多样性。

- 添加正则化项。

- 增加模型的宽度和深度，或者减少层数。

- 早停法。通过条件语句判断loss是否继续下降。

- 数据增强（翻转、平移、缩放）。

- Dropout

- 调整超参（LR、Batchsize等）。

  

## 构建网络<div id='net'></div>

```python
Import pkgs

# define config
parser = argparse.ArgumentParser()
parser.add_argument("--xx", type, default)
args = parser,parse_args()
# cpu or gpu
DEVICE = 'gpu' if torch.cuda.is_available()  else 'cpu'

# initiate weights
def init_weights():
 
# define net
Class Network(nn.module):
 
# Load dataset

model = Network().to(DEVICE)
model.apply(init_weights)

# define opt and Loss

# train

# save model
```

## 损失函数<div id='loss'></div>

- 二分类
	- BCE Loss
- 多标签分类
  - CE Loss = LogSoftmax + NLL Loss
  - weighted-CE Loss
  - NLL Loss 如果最后一层用的是LogSoftmax

- 回归
  - L1 Loss
  - MSE Loss 对异常点敏感
  - SmoothL1 Loss 平滑版L1Loss, 防止梯度爆炸

## 训练<div id='train'></div>

- 小样本训练：
  - 数据增强
  - 迁徙学习，使用一个预训练模型，然后进行微调，但是预训练模型使用的数据集最好与新数据集有相似的分布。
  - 正则化
  - 使用更小的网络，更少的参数防止过拟合

- 大样本训练：
  - 分批次训练
  - 使用适合大数据的优化器