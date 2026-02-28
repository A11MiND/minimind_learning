import torch

# # torch.where(condition, x, y)函数根据条件condition选择x或y中的元素，返回一个新的张量result。
# # condition是一个布尔张量，表示每个元素是否满足条件。
# # 如果condition中的元素为True，则result中的对应元素取自x；如果为False，则取自y。
# x = torch.tensor([1, 2, 3, 4, 5])
# y = torch.tensor([10, 11, 12, 13, 14])

# condition = x > 3

# result = torch.where(condition, x, y)

# print(result)

# # 创建一个从0到10，步长为2的张量
# t = torch.arange(0, 10, 2)
# print(t)

# # 创建一个从5到1，步长为-1的张量
# t2 = torch.arange(5, 0, -1)
# print(t2)

# # 外积（Outer Product）是线性代数中的一个操作，它将两个向量作为输入，生成一个矩阵。对于两个向量v1和v2，外积的结果是一个矩阵，其中第i行第j列的元素等于v1[i] * v2[j]。
# v1 = torch.tensor([1, 2, 3])
# v2 = torch.tensor([4, 5, 6])
# result = torch.outer(v1, v2)
# print(result)
# # tensor([[ 4,  5,  6],
#         [ 8, 10, 12],
#         [12, 15, 18]])

# # cat函数用于连接两个或多个张量。它接受一个张量列表和一个维度参数dim，表示沿着哪个维度进行连接。连接后的张量的形状取决于输入张量的形状和连接的维度。
# t1 = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]])
# t2 = torch.tensor([[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]])
# # result = torch.cat((t1, t2), dim=0)
# result = torch.cat((t1, t2), dim=-1)
# print(t1.shape) # torch.Size([2, 2, 3])
# print(t2.shape) # torch.Size([2, 2, 3])
# print(result.shape) # torch.Size([4, 2, 3])
# print(result)

# t1 = torch.Tensor([1, 2, 3])
# t2 = t1.unsqueeze(0)
# print(t1.shape) # torch.Size([3])
# print(t2.shape) # torch.Size([1, 3])
# print(t2) # tensor([[1., 2., 3.]])







