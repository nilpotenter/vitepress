# 深度学习基础

## 【了解】今日内容介绍

* PyTorch的API（续）
* 自动微分模块
* 线性回归案例



## 【实现】PyTorch使用

### 什么是张量

深度学习中，为什么会引入张量，就是因为它支持GPU，提升计算效率。

张量，本质就是一个数组，是以类的形式封装起来。

PyTorch张量与NumPy数组类似，但PyTorch的张量具有GPU加速的能力（通过CUDA），这使得深度学习模型能够高效地
在GPU上运行。

PyTorch提供了对张量的强大支持，可以进行高效的数值计算、矩阵操作、自动求导等。

按照数据维度不同，张量也可以分为一维张量、二维张量、三维张量、更高维度的张量等。



### 张量的创建

#### 基本创建方式

- torch.tensor 根据指定数据创建张量
- torch.Tensor 根据形状创建张量, 其也可用来创建指定数据的张量
- torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量

代码如下：

```python
import numpy as np
import torch

# 创建张量标量
data = torch.tensor(10)
print(data)

# 创建张量数组
data = np.random.randn(3, 2)
print(data)
print(data.dtype)
print(torch.tensor(data))

# 创建张量列表
data = [[10., 20., 30.], [40., 50., 60]]
data = torch.tensor(data)
print(data, data.dtype)
```

> 这种方式常用。



```python
# 根据形状
data = torch.Tensor(3)
print(data)
data = torch.Tensor(3, 2)
print(data)
data = torch.Tensor([3, 2])
print(data)
```

> 这种方式用的不多。



```python
# 指定类型
data = torch.IntTensor(3)
print(data)
data = torch.IntTensor(3, 2)
print(data)
# 如果类型不对，还可以进行转换
data = torch.IntTensor([3.2, 2.5])
print(data)
data = torch.FloatTensor(2)
print(data)
data = torch.DoubleTensor(2)
print(data)
```

> 这种方式用的不多。



#### 创建线性和随机张量

- torch.arange 和 torch.linspace 创建线性张量

```python
# [0,3)之间，按照step=1，一共产出3个元素
# 核心：step是不变的
data = torch.arange(start=0, end=3, step=1)
print(data, data.dtype)

# [0,3]之间，按照steps=2，生成2个元素
# 核心：steps是不变的
data = torch.linspace(start=0, end=3, steps=2)
print(data, data.dtype)
```



- torch.random.init_seed 和 torch.random.manual_seed 随机种子设置
- torch.randn 创建随机张量

```python
# 手动设置随机种子
torch.random.manual_seed(22)
data = torch.randn(3)
print(data)
# 获取随机种子
print(torch.random.initial_seed())

data = torch.randn(2, 3)
print(data)
```



#### 创建0-1张量

- torch.ones 和 torch.ones_like 创建全1张量
- torch.zeros 和 torch.zeros_like 创建全0张量
- torch.full 和 torch.full_like 创建全为指定值张量

```python
# ones和ones_like
data = torch.ones(2, 3)
print(data)
data = torch.ones_like(data)
print(data)

# zeros和zeros_like
data = torch.zeros(3, 2)
print(data)
data = torch.zeros_like(data)
print(data)

# full和full_like
data = torch.full((2, 3), 3)
print(data)
data = torch.full_like(data, fill_value=5)
print(data)
```



#### 张量类型转换

- data.type(torch.DoubleTensor)
- data.double()

```python
# 转换为short类型
data = torch.randn(2, 3)
print(data, data.dtype)
# data = data.type(torch.ShortTensor)
data = data.type(torch.LongTensor)
print(data)

# 转换为double类型
data = data.double()
# data = data.short()
# data = data.int()
# data = data.long()
# data = data.float()
print(data)
```

PyTorch中的数据类型如下：

| 数据类型                   | 位数       |
| -------------------------- | ---------- |
| torch.float/torch.float32  | 32位浮点型 |
| torch.double/torch.float64 | 64位浮点型 |
| torch.int8                 | 8位整型    |
| torch.int16/torch.short    | 16位整型   |
| torch.int32/torch.int      | 32位整型   |
| torch.int64/torch.long     | 64位整型   |
|                            |            |



### 张量类型转换

#### 张量转换为数组

- data.numpy()

使用Tensor.numpy()函数可以将张量转换为ndarray数组，但是共享内存，可以使用copy()函数避免共享

```python
import torch

# 张量转换为数组
data_tensor = torch.tensor([1, 2, 3, 4])
print(data_tensor, type(data_tensor))
# copy()方法，可以避免内存共享
data_numpy = data_tensor.numpy()
print(data_numpy, type(data_numpy))
# 注意: data_tensor 和 data_numpy 共享内存
# 修改其中的一个，另外一个也会发生改变
# 修改data_numpy中某个元素的值，data_tensor也会改
data_numpy[-1] = 400
print(data_numpy)
print(data_tensor)
```

#### 数组转换为张量

- 使用 from_numpy 可以将 ndarray 数组转换为 Tensor，默认共享内存，使用 copy 函数避免共享。
- 使用 torch.tensor 可以将 ndarray 数组转换为 Tensor，默认不共享内存。

```python
import torch
import numpy as np

# 准备一个numpy数据
data = np.array([[1, 2], [3, 4]])
print(data, type(data))
# 把numpy数据转换为张量
data1 = torch.from_numpy(data)
# copy，避免共享
data1 = torch.from_numpy(data.copy())
print(data1, type(data1))
# 修改numpy的数据
data[0][0] = 100
print(data)
# 查看张量的数据是否会被影响
print(data1)
```

#### 标量张量和数字的转换

- 对于只有一个元素的张量，使用 item 方法将该值从张量中提取出来。

```python
data = torch.tensor([30])
print(data)
print(data.item())
```

如果张量超过一个元素，则会报错：

`ValueError: only one element tensors can be converted to Python scalars`



### 张量数值计算

#### 张量基本运算

加、减、乘、除、取负号：add、sub、mul、div、neg 

`add_、sub_、mul_、div_、neg_`（其中带下划线的版本会修改原数据）

```python
import torch

data = torch.randint(low=0, high=10, size=(2, 3))
print(data, data.dtype)

print(data.add(1))

print(data.sub(1))

print(data.mul(2))

print(data.div(2))

print(data.neg())

# 下划线会修改原数据
# data.add_(2)
data.add(2)
print(data)

```

#### 矩阵的点乘

点乘指（Hadamard）的是两个同维矩阵对应位置的元素相乘，使用`mul`或者运算符`*`实现。

```python
data1 = torch.tensor([[1, 2], [3, 4]])
data2 = torch.tensor([[5, 6], [7, 8]])
print(data1)
print(data2)
# 方式一
print(data1 * data2)
# 方式二
print(data1.mul(data2))
```

> Tips：同维度矩阵并不是指两个矩阵的维度要一样。



#### 矩阵乘积运算

矩阵运算要求第一个矩阵 shape: (n, m)，第二个矩阵 shape: (m, p), 两个矩阵点积运算 shape 为: (n, p)。

```python
# data1 = torch.tensor([[1, 2], [3, 4], [5, 6]]) # (3, 2)
data1 = torch.tensor([[[1, 2], [3, 4], [5, 6]]]) # (1, 3, 2)
# data2 = torch.tensor([[5, 6], [7, 8]]) # (2, 2)
# data2 = torch.tensor([[[5, 6], [7, 8]]]) # (1, 2, 2)
data2 = torch.tensor([[[5, 6], [7, 8]], [[5, 6], [7, 8]]]) # (2, 2, 2)

# 矩阵乘法的简写：@
# print(data1 @ data2)

# mm：只能是二维数据相乘
# print(torch.mm(data1, data2))

# bmm：需要3维数据，且第一维要相同
# print(torch.bmm(data1, data2))

# matmul，可以是二维，也可以是三维，
print(torch.matmul(data1, data2))
```

- 运算符 @ 用于进行两个矩阵的乘积运算
- torch.mm 用于进行两个矩阵乘积运算, 要求输入的矩阵为2维
- torch.bmm 用于批量进行矩阵乘积运算, 要求输入的矩阵为3维
- torch.matmul 对进行乘积运算的两矩阵形状没有限定
  - 对于输入都是二维的张量相当于 mm 运算
  - 对于输入都是三维的张量相当于 bmm 运算
  - 对数输入的 shape 不同的张量, 对应的最后几个维度必须符合矩阵运算规则



### 张量运算函数

PyTorch 为每个张量封装很多实用的计算函数，例如计算均值、平方根、求和、指数、对数等等。

```python
import torch
torch.set_printoptions(sci_mode=False)

# 要转换为float或者double类型，否则会报错
data = torch.randint(0, 10, (3, 2), dtype=torch.float32)
print(data, data.dtype)

# 均值：总数加起来/总个数
print(data.mean())
print(data.mean(dim=0))  # 按列计算均值
print(data.mean(dim=1))  # 按行计算均值

# 平方根：每个数都求平方根
print(data.sqrt())

# 求和：所有数加起来
print(data.sum())
print(data.sum(dim=0))
print(data.sum(dim=1))

# 指数计算：每个元素进行e^x计算
print(data.exp())

# 对数计算：以e为底数，增函数
print(data.log())
print(data.log2())
print(data.log10())
```



### 张量索引操作

我们在操作张量时，经常需要去获取某些元素就进行处理或者修改操作。

#### 简单行、列索引

```python
import torch

# 准备张量
torch.random.manual_seed(22)
data = torch.randint(0, 10, [4, 5])
print(data)

# 简单行、列索引
print(data[2]) # 索引行
print(data[:, 2]) # 索引列
```

> 这种常用。

#### 列表索引

```python
import torch

# 准备张量
torch.random.manual_seed(22)
data = torch.randint(0, 10, [4, 5])
print(data)


# 列表索引
# 返回（0,1）和（1,2）两个位置的元素
print(data[[0, 1], [1, 2]])
# 返回0、1行的1、2列共4个元素
print(data[[[0], [1]], [1, 2]])
```



#### 范围索引

```python
import torch

# 准备张量
torch.random.manual_seed(22)
data = torch.randint(0, 10, [4, 5])
print(data)

# 范围索引，左闭右开区间
print(data[: 2])
print(data[:, :2])
print(data[:, :])
print(data[:3, :2])
print(data[2:, :2])
```

> 这种常用。

#### 布尔索引

```python
import torch

# 准备张量
torch.random.manual_seed(22)
data = torch.randint(0, 10, [4, 5])
print(data)

# 布尔索引
# 过滤行
print(data[data[:, 2] > 5])
# 过滤列
print(data[:, data[1] > 5])
```



#### 多维索引

```python
import torch

# 多维索引
torch.random.manual_seed(22)
data = torch.randint(0, 10, [3, 4, 5])
print(data)

# 获取第1个二维数组
print(data[0, :, :])
# 获取每个二维数组中的第一行
print(data[:, 0, :])
# 获取每个一维数组中的第一列
print(data[:, :, 0])
```

小结：行列索引和范围索引常用。



### 张量形状操作

我们后面搭建网络模型时，数据都是基于张量形式的表示，网络层与层之间很多都是以不同的 shape 的方式进行表现和运算，我们需要掌握对张量形状的操作，以便能够更好处理网络各层之间的数据连接。

#### reshape函数

reshape函数可以在保证张量数据不变的前提下改变数据的维度，将其转换成指定的形状。 

```python
import torch

# shape和size()是一样的
data = torch.tensor([[10, 20, 30], [40, 50, 60]])
print(data, data.dtype)
print(data.shape, data.size())
print(data.shape[1], data.size(1))

# reshape可以改变数据的形状
# data = data.reshape(3, 2)
data = data.reshape(3, -1)
print(data)
```



#### squeeze和unsqueeze函数

squeeze 函数删除 形状为 1 的维度（升维），unsqueeze 函数添加形状为1的维度（降维）。

```python
import torch

data = torch.tensor([[10, 20, 30], [40, 50, 60]])
print(data.shape)
# unsqueeze(dim)：需要在哪个维度升维
data = data.unsqueeze(-1)
print(data.shape)
# squeeze(dim)：需要在哪个维度降维（必须为1）或者省略（压缩维度）
data = data.squeeze()
print(data.shape)
```



#### transpose和permute函数

transpose 函数可以实现交换张量形状的指定维度, 例如: 一个张量的形状为 (2, 3, 4) 可以通过 transpose 函数把 3 和 4 进行交换, 将张量的形状变为 (2, 4, 3)

permute 函数可以一次交换更多的维度。

```python
data = torch.randint(0, 10, (3, 4, 5))
print(data, data.shape) # [3,4,5]

data = data.transpose(0, 1)
print(data, data.shape) # [4,3,5]

data = data.permute(2, 0, 1)
print(data, data.shape) # [5,4,3]
```



#### view和contiguous函数

view 函数也可以用于修改张量的形状，但是其用法比较局限，只能用于存储在整块内存中的张量。在 PyTorch 中，有些张量是由不同的数据块组成的，它们并没有存储在整块的内存中，view 函数无法对这样的张量进行变形处理，例如: 一个张量经过了 transpose 函数的处理之后，就无法使用 view 函数进行形状操作。

```python
import torch

data = torch.randint(0, 10, (3, 4, 5))
print(data, data.shape)

data = data.transpose(1, 2) # 不连续
# data = data.permute(1, 2, 0)
# data = data.view(1, 2, -1) # data不连续后，调用view函数会报错

print(data, data.shape)
print(data.is_contiguous()) # 判断是否连续
print(data.contiguous().is_contiguous()) # 通过contiguous把不连续的内存空间变成连续
```

结论：

- reshape 函数可以在保证张量数据不变的前提下改变数据的维度。
- squeeze 和 unsqueeze 函数可以用来增加或者减少维度。
- transpose 函数可以实现交换张量形状的指定维度，permute 可以一次交换更多的维度。
- view 函数也可以用于修改张量的形状, 但是它要求被转换的张量内存必须连续，所以一般配合 contiguous 函数使用。



### 张量拼接操作

张量的拼接操作在神经网络搭建过程中是非常常用的方法，例如: 在后面将要学习的注意力机制中都使用到了张量拼接。

torch.cat 函数可以将两个张量根据指定的维度拼接起来，不改变数据维度。

前提：除了拼接的维度，其他维度一定要相同。

```python
import torch

data1 = torch.randint(0, 10, (1, 2, 3))
print(data1, data1.shape)
data2 = torch.randint(0, 10, (1, 2, 3))
print(data2, data2.shape)
# cat拼接
data = torch.cat(tensors=[data1, data2], dim=0)
print(data, data.shape)
# concat拼接
data = torch.concat(tensors=[data1, data2], dim=0)
print(data, data.shape)
# concatenate拼接
data = torch.concatenate(tensors=[data1, data2], dim=0)
print(data, data.shape)
```

结论：

- cat、concat、concatenate三者效果一样，优先使用cat拼接。



### 自动微分模块

我们知道，无论是机器学习还是深度学习，都有损失函数的概念。

如果要对损失函数进行优化，优化方法有正规方程和梯度下降。

随着模型越来越复杂，一般都是使用梯度下降。

自动微分，就是pytorch框架在梯度下降的一种实现方式。

在pytorch中，要想使用梯度下降法来寻得最优解，就得使用自动微分模块来实现。

一句话，在PyTorch中，我们使用自动微分模块来计算导数（梯度）。

pytorch对自动微分封装的已经很完美了。就是一个方法：backward()。



训练神经网络时，最常用的算法就是反向传播。在该算法中，参数（模型权重）会根据损失函数关于对应参数的梯度进行调整。为了计算这些梯度，PyTorch内置了名为 torch.autograd 的微分引擎。它支持任意计算图的自动梯度计算：

![](https://imgbed.nilpo.ddns-ip.net/20250817164212139.png)

我们使用`backward`方法、`grad`属性来实现梯度的计算和访问。

需要计算梯度的张量需要设置`requires_grad=True` 属性。 

梯度计算代码如下：

```python
import torch

def test01():
    x = torch.tensor([3.], requires_grad=True)
    # x = torch.tensor([[3, 4, 5.], [1, 2, 6.]], requires_grad=True)

    y = x ** 2  # 2x

    # z = 2 * y + 3 #

    print('x-->', x.grad) # None，第一次，还没有计算梯度，梯度为None

    y.sum().backward()
    # z.sum().backward()

    print('x-->', x.grad) # 自动微分后，于是有了梯度，输出才有值


def test02():
    x = torch.tensor([3.], requires_grad=True)
    # y = x ** 2

    for i in range(100):
        y = x ** 2

        # 如果不清零，梯度会累加计算
        if x.grad is not None:
            x.grad.zero_()

        y.sum().backward()

        x.data = x.data - 0.01 * x.grad # w1 = w0 - lr * grad

        print(x.grad)


if __name__ == '__main__':
    test01()
    # test02()
```

完整流程代码如下：

```python
def demo03():
    # 准备x、w、b、z
    x = torch.tensor(3.)
    w = torch.tensor(2., requires_grad=True)
    b = torch.tensor(1., requires_grad=True)
    z = w * x + b
    # 准备y
    y = torch.tensor(0.)
    
    # w:(x * w + b - y)^2 = 2(x * w + b - y)*x = 2(2*3+1)*3 = 14*3 = 42
    # b:(x * w + b - y)^2 = 2(x * w + b - y)*1 = 2(2*3+1)*1 = 14*1 = 14
    # 计算损失
    criterion = nn.MSELoss()
    loss = criterion(z, y)
    # 使用自动微分模块计算梯度
    loss.backward()
    # 查看梯度结果
    print('w-->', w.grad)
    print('b-->', b.grad)


def demo04():
    # 准备x、w、b、z
    x = torch.randn(2, 3)
    w = torch.randn(3, 5, requires_grad=True)
    b = torch.randn(5, requires_grad=True) # 广播机制：(1,5) -> (2,5)，实现和y的相加
    z = torch.matmul(x, w) + b
    # 准备y
    y = torch.randn(2, 5)
    # 计算损失
    criterion = nn.MSELoss()
    loss = criterion(z, y)
    # 使用自动微分模块计算梯度
    loss.backward()
    # 查看梯度结果
    print('w-->', w.grad)
    print('b-->', b.grad)

```

小结（梯度的理解）：

```shell
#1.数值上的理解
在某一个点上，对函数求导得到的值就是梯度，没有点谈梯度，没有任何意义

#2.方向上的理解
梯度就是上山下山最快的方向

#3.斜率的理解
平面内，梯度就是某一点上的斜率

#4.反向传播的理解
反向传播，传播的是梯度
因为反向传播，利用链式法则，从后向前求导，求出来的值就是梯度，所以大家经常说反向传播，传播的是梯度
前向传播，传播的是模型的输出结果。

#5.链式法则的理解
链式法则中，梯度相乘，就是传说说的梯度传播
```



### 线性回归案例

#### 流程

今日线性回归案例的大致流程如下：

![](https://imgbed.nilpo.ddns-ip.net/20250817164312459.png)

- 使用 PyTorch 的 data.DataLoader 代替自定义的数据加载器
- 使用 PyTorch 的 nn.Linear 代替自定义的假设函数
- 使用 PyTorch 的 nn.MSELoss() 代替自定义的平方损失函数
- 使用 PyTorch 的 optim.SGD 代替自定义的优化器



#### 套路

~~~shell
#1.数据处理的套路
文本数值化 -> 数值张量化

#2.模型训练的套路
数据 -> 先转换为Dataset -> 转换为DataLoader
~~~



#### 实现

```python
import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
import warnings
from torch.utils.data import DataLoader, TensorDataset
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

warnings.filterwarnings(action="ignore")


#1.构造数据集
def get_dataset():
    x, y, coef = make_regression(n_samples=100, n_features=1, bias=1.5, noise=10, coef=True, random_state=22)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y, coef


#2.可视化数据集
def show_dataset():
    x, y, coef = get_dataset()
    plt.scatter(x, y)
    plt.plot(x, x * coef + 1.5)
    plt.grid()
    plt.show()


#3.构建模型
def make_model():
    # 模型
    model = nn.Linear(in_features=1, out_features=1)
    # 损失函数
    loss_fn = nn.MSELoss()
    # 优化器
    optimizer = optim.SGD(params=model.parameters(), lr=0.01)
    return model, loss_fn, optimizer


#4.模型训练
def train_model():
    # 获取数据迭代器
    x, y, coef = get_dataset()
    x1 = x
    y1 = y

    # 转换为dataloader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
    # 获取模型
    model, loss_fn, optimizer = make_model()
    # 准备变量
    epochs = 100
    total_sample = 0
    total_loss = 0
    epoch_loss = []
    # 模型训练
    # 外层循环控制轮次
    for epoch in range(epochs):
        # 内存循环控制批次
        for x, y in dataloader:
            # 模型预测
            y_pred = model(x)
            # 计算损失，因为y_pred：[8,1],所以需要把y也变成[8,1]的形状
            loss = loss_fn(y_pred, y.reshape(-1, 1))
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 纪录数据
            total_loss += loss.item()
            total_sample += len(y)

        epoch_loss.append(total_loss / total_sample)

        # 打印日志
        print("当前轮次：",(epoch + 1), "当前平均损失：", total_loss/total_sample)

    # 绘制损失曲线
    plt.plot(range(epochs), epoch_loss)
    plt.grid()
    plt.title("损失变化曲线")
    plt.show()

    # 绘制拟合曲线
    x1 = x1.detach().numpy()
    y1 = y1.detach().numpy()
    plt.scatter(x1, y1)
    plt.plot(x1, x1 * coef + 1.5, label='true')
    plt.plot(x1, model.weight.detach().numpy() * x1 + model.bias.detach().numpy(), label='train')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # show_dataset()
    # make_model()
    train_model()
```
