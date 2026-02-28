import torch
import torch.nn as nn


# # Dropout层是神经网络中的一种正则化技术，用于防止过拟合。
# 它通过在训练过程中随机将一部分神经元的输出设置为零来实现。
# 这有助于模型更好地泛化到未见过的数据。

# dropout_layer = nn.Dropout(p=0.4)

# t1 = torch.Tensor([[1.0, 2.0, 3.0]])
# t2 = dropout_layer(t1)
# print(t2)

# # 线性变换,对应的张量乘以一个w矩阵并加上一个b向量
# layer = nn.Linear(in_features=3, out_features=5, bias=True)
# t1 = torch.Tensor([1, 2, 3]) #shape: torch.Size([3])
# t2 = torch.Tensor([[4, 5, 6]]) #shape: torch.Size([1, 3])

# output2 = layer(t2)
# print(output2)

# # view函数用于改变张量的形状。它接受一个新的形状作为参数，并返回一个新的张量，该张量与原始张量共享数据，但具有不同的形状。
# t = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]) #shape: torch.Size([2, 6])
# t_view1 = t.view(3, 4)
# print(t_view1) #shape: torch.Size([3, 4])
# t_view2 = t.view(4, 3)
# print(t_view2) #shape: torch.Size([4, 3])

# # transpose函数用于交换张量的维度。它接受两个维度作为参数，并返回一个新的张量，其中指定的维度被交换。
# t1 = torch.Tensor([[1, 2, 3], [4, 5, 6]])
# t1 = t1.transpose(0, 1)
# print(t1)

# 掩码计算
# x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(torch.triu(x, diagonal=0)) # 上三角矩阵，diagonal=0表示主对角线，diagonal=1表示对角线往上移，diagonal=-1表示对角线往下移
# print(torch.tril(x, diagonal=0)) # 下三角矩阵，diagonal

# # reshape函数用于改变张量的形状。它接受一个新的形状作为参数，并返回一个新的张量，该张量与原始张量共享数据，但具有不同的形状。
# # 和view函数类似，但reshape在某些情况下可能会返回一个新的张量，而view总是返回一个视图（共享数据的张量）。因此，reshape更灵活，可以处理一些特殊情况。
# 从内存上，view要求输入张量是连续的，而reshape不要求输入张量是连续的。
# x = torch.arange(1, 7) # shape: torch.Size([6])
# y = torch.reshape(x, (2, 3))
# z = torch.reshape(x, (3, -1)) #-1表示自动计算维度大小
# print(y)
# print(z)


