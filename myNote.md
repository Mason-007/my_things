# Contents:

- [Overfitting](#ovft)
- [How to train a CNN](#train)

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

  

## 训练网络<div id='train'></div>

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

