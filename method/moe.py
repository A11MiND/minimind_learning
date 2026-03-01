import torch

# torch.div 张量的逐元素除法
# output = import/ other

# a = torch.tensor([10, 20, 30])
# b = torch.tensor([10, 10, 10])
# ans = torch.div(a, b)
# print(ans)

# # torch.mean 对张量求平均值
# x = torch.tensor([1., 2., 3.])
# y = torch.tensor([[1., 2., 3.],[4., 5., 6.]])
# c = torch.mean(y, dim=0) #对行求平均值
# d = torch.mean(y, dim=1) #对列求平均值
# print(c)
# print(d)

# # torch.scatter_add_ 按照索引把值驾到目标张量指定位置
# # 把东西按index分桶加进去
# out = torch.zeros(5)
# index = torch.tensor([0, 1, 0, 3])
# src = torch.tensor([1., 2., 3., 4.])
# out.scatter_add_(dim=0, index=index, src=src)
# print(out)


