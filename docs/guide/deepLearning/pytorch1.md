

## PyTorch入门：张量（Tensor）的奇妙世界

欢迎来到PyTorch的世界！如果你是深度学习的新手，或者正从其他框架（如NumPy）迁移过来，那么你首先需要掌握的核心概念就是——**张量（Tensor）**。

在PyTorch中，张量是所有运算的基础数据结构。你可以把它想象成一个多维数组，类似于NumPy中的`ndarray`。实际上，它们可以无缝衔接。张量是构建神经网络、执行梯度计算的基石。

### 1. 万物之始：创建张量

创建张量的方法多种多样，可以根据已有的数据创建，也可以指定形状来创建。

#### 1.1 根据已有数据创建

这是最直接的方式，使用`torch.tensor()`函数，它可以接收Python列表、NumPy数组等多种数据类型。

```python
import torch
import numpy as np

# 从Python列表创建一维张量
data1 = torch.tensor([1, 2, 3])
print(data1, data1.dtype)
# 输出: tensor([1, 2, 3]) torch.int64

# 从一个单独的数字（标量）创建零维张量
data1 = torch.tensor(3)
print(data1, data1.dtype)
# 输出: tensor(3) torch.int64

# 从NumPy数组创建，并指定数据类型为float
numpy_array = np.random.randn(3, 4)
data1 = torch.tensor(numpy_array, dtype=torch.float32) # 推荐使用torch.float32或torch.float
print(data1, data1.dtype)
# 输出: (一个3x4的随机张量) torch.float32
```

**注意**：`torch.tensor()`会复制数据。这是最安全、最推荐的创建方式。

#### 1.2 根据形状创建

有时候我们想先初始化一个特定形状和类型的张量，然后再填充数据。

```python
# 注意：torch.Tensor() 使用全局默认类型（通常是float32），并用未初始化的数据填充
# 为了代码的可预测性，不推荐直接使用，除非你清楚你在做什么
data_shape = torch.rand(2, 3) # 先创建一个有确定值的张量
data1 = torch.Tensor(data_shape)
print(data1, data1.dtype)
# 输出: (一个2x3的张量，值与data_shape相同) torch.float32

# 也可以使用更明确的构造函数
data1 = torch.FloatTensor(2, 3) # 创建一个2x3的浮点型张量（未初始化）
print(data1, data1.dtype)

data1 = torch.IntTensor(2, 3)   # 创建一个2x3的整型张量（未初始化）
print(data1, data1.dtype)
```

#### 1.3 创建特殊张量

PyTorch提供了丰富的函数来创建具有特定值的张量，这在初始化权重或构建模型时非常有用。

##### 线性与序列张量

```python
# 创建一个从0到9，包含10个元素的等差数列张量
data1 = torch.linspace(0, 9, 10)
print(data1, data1.dtype)
# 输出: tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]) torch.float32

# 创建一个从0到8（不含9），步长为2的整数序列张量
data1 = torch.arange(0, 9, 2)
print(data1, data1.dtype)
# 输出: tensor([0, 2, 4, 6, 8]) torch.int64
```

##### 随机张量

```python
# 产生均值为0，标准差为1的正态分布张量（标准正态分布）
data1 = torch.randn(size=(2, 3))
print(data1, data1.dtype)

# 产生在[0, 1)区间上均匀分布的随机张量
data1 = torch.rand(size=(2, 3))
print(data1, data1.dtype)

# 产生在[1, 5)区间上的随机整数张量
data1 = torch.randint(1, 5, (3, 5))
print(data1, data1.dtype)
```

##### 全0，全1，或全n张量

```python
# 创建全0张量
data1 = torch.zeros(3, 3)
print(data1, data1.dtype)

# 创建全1张量
data1 = torch.ones(3, 3)
print(data1, data1.dtype)

# 创建一个形状为(3,3)，所有元素都为4的张量
# 注意：函数是 torch.full((shape), value)
data1 = torch.full((3, 3), 4)
print(data1, data1.dtype)

# _like系列函数：根据另一个张量的形状、类型和设备创建新张量，非常方便
template_tensor = torch.rand(2, 4)
zeros_like_tensor = torch.zeros_like(template_tensor)
print(zeros_like_tensor, zeros_like_tensor.shape) # 形状与template_tensor一致

ones_like_tensor = torch.ones_like(template_tensor)
print(ones_like_tensor, ones_like_tensor.shape)
```
**注意**：原文中的`torch.zeros_like(3,3)`是错误的用法。`_like`函数需要传入一个已存在的张量作为模板，而不是形状参数。

### 2. 张量的属性与转换

创建张量后，我们经常需要改变它的类型或在PyTorch与NumPy之间来回转换。

#### 2.1 数据类型转换
有多种方式可以改变张量的数据类型。

```python
# 假设我们有一个浮点张量
float_tensor = torch.full((2, 2), 4.0)
print(float_tensor, float_tensor.dtype)

# 方法1: 使用.type()方法，最灵活
int_tensor = float_tensor.type(torch.IntTensor) # 或者 torch.int
print(int_tensor, int_tensor.dtype)

# 方法2: 使用便捷方法，如 .int(), .float(), .double()
int_tensor_again = float_tensor.int()
print(int_tensor_again, int_tensor_again.dtype)
```

#### 2.2 与NumPy的无缝衔接

PyTorch与NumPy的互操作性是其一大亮点。

```python
# 张量转NumPy
tensor_to_convert = torch.ones(2, 3)
numpy_array = tensor_to_convert.numpy()
print(f"Numpy array type: {type(numpy_array)}")
# 注意：转换后的Tensor和ndarray共享内存！修改一个会影响另一个。
# 如果不希望共享内存，请使用copy()
numpy_array_copy = tensor_to_convert.numpy().copy()

# NumPy转张量
numpy_to_convert = np.random.randn(2, 3)

# 方法1: torch.from_numpy()，同样共享内存
tensor_from_numpy = torch.from_numpy(numpy_to_convert)
print(f"Tensor type: {type(tensor_from_numpy)}")

# 方法2: torch.tensor()，不共享内存（推荐）
tensor_copy = torch.tensor(numpy_to_convert)
print(f"Tensor type: {type(tensor_copy)}")
```
**核心要点：** `torch.from_numpy()`和`.numpy()`会共享内存，这是一个高效的特性，但也可能导致意外的错误。如果想避免这种情况，请使用`.copy()`或`torch.tensor()`。

#### 2.3 从标量张量到Python数字

当你的张量只有一个元素时，可以使用`.item()`方法将其提取为标准的Python数字。

```python
scalar_tensor = torch.tensor(42)
num = scalar_tensor.item()
print(num, type(num))
# 输出: 42 <class 'int'>
```
这在处理损失函数值或评估指标时非常有用。

### 3. 张量的核心：运算

张量支持丰富的数学运算，这是构建模型的基础。

#### 3.1 逐元素运算

这些运算作用于张量的每一个元素。

```python
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
b = torch.ones(2, 2)

# 基本算术
print("加法:", a.add(3))      # 或 a + 3
print("减法:", a.sub(3))      # 或 a - 3
print("乘法:", a.mul(3))      # 或 a * 3
print("除法:", a.div(3))      # 或 a / 3
print("取负:", a.neg())       # 或 -a
```

#### 3.2 In-place操作

在PyTorch中，任何以`_`结尾的操作（如`add_`）都是**in-place**（原地）操作，它会直接修改张量自身，而不是返回一个新张量。

```python
print("Original a:", a)
a.add_(3) # a的值被直接修改
print("After add_:", a)
```
原地操作可以节省内存，但在计算梯度时可能会有问题，使用时需要小心。

#### 3.3 矩阵乘法

这是深度学习中最常见的运算之一。千万不要和逐元素乘法（`*`或`mul`）混淆！

```python
data1 = torch.randn(2, 3)
data2 = torch.randn(3, 4)

# 逐元素乘法 (Hadamard Product) - 需要形状能广播匹配
# tensor1 = torch.tensor([[1,2],[3,4]])
# tensor2 = torch.tensor([[5,6],[7,8]])
# print(tensor1.mul(tensor2)) # 输出 tensor([[5, 12], [21, 32]])

# 矩阵乘法
# 方法1: 使用@符号 (Python 3.5+推荐)
result1 = data1 @ data2
print("Result with @:\n", result1)
print("Shape:", result1.shape) # 形状为 (2, 4)

# 方法2: 使用 torch.matmul()
result2 = torch.matmul(data1, data2)
print("Result with matmul:\n", result2)

# 方法3: 使用 .mm()，仅适用于2D矩阵
result3 = data1.mm(data2)
print("Result with mm:\n", result3)

# 批量矩阵乘法 (bmm)
# 假设我们有两个batch的矩阵，每个batch包含一个 2x2 矩阵
batch1 = torch.randn(5, 2, 3) # Batch size=5, 矩阵为2x3
batch2 = torch.randn(5, 3, 4) # Batch size=5, 矩阵为3x4
batch_result = torch.bmm(batch1, batch2)
print("Batch mm result shape:", batch_result.shape) # (5, 2, 4)
```

### 附录：Python虚拟环境管理（Conda）

在进行深度学习项目时，良好的环境管理至关重要。Conda是一个优秀工具。

```bash
# 查看所有已创建的虚拟环境
conda env list

# 创建一个名为pytorch，python版本为3.8的新环境
conda create -n pytorch python=3.8

# 激活环境
conda activate pytorch

# (在环境中工作...)
# pip/conda install pytorch torchvision torchaudio -c pytorch

# 查看当前环境已安装的包
pip list
conda list

# 导出环境配置，方便他人复现
pip freeze > requirements.txt

# 从配置文件安装所有依赖
pip install -r requirements.txt

# 完成工作后，退出环境
conda deactivate

# 如果不再需要，可以删除整个环境
conda remove -n pytorch --all
```

### 总结

今天，我们探索了PyTorch的核心——张量。我们学习了如何：
- 从各种数据源**创建**张量。
- **转换**张量的数据类型，以及与NumPy的交互。
- 执行基本的**数学运算**，特别是区分了逐元素乘法和矩阵乘法。

掌握张量是通往PyTorch proficiency的第一步，也是最重要的一步。希望这篇指南对你有所帮助！接下来，你可以继续探索张量的索引、切片、变形（reshape）以及自动求导（Autograd）系统。祝你学习愉快！