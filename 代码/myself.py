import  numpy as np
import  matplotlib.pyplot as plt
import  math
import  pandas as pd
import math

 #scatter用法，有size（点的大小）和color参数
# x = np.random.rand(200)
# y = np.random.rand(200)
# size = np.random.rand(200) * 300
# color = np.random.rand(200)
# plt.scatter(x, y, size, color)
# # 显示颜色条
# plt.colorbar()

#可以在 plot 中加入 label ，使用 legend 加上图例：
# plt.plot(x, label='sin')
# plt.plot(y, label='cos')
# plt.legend()

# 清除、关闭图像
# 清除已有的图像使用：
# clf()
# 关闭当前图像：
# close()
# 关闭所有图像：
# close('all')

# dir() 函数
# dir() 函数不带参数时，返回当前范围内的变量、方法和定义的类型列表；带参数时，返回参数的属性、方法列表。如果参数包含方法__dir__()，
# 该方法将被调用。如果参数不包含__dir__()，该方法将最大限度地收集参数信息。
#可以用这个函数查看如何使用，和属性

#numpy里，ndarray.size是元素数目；ndarray.shape是形状
#ndarray.ndim是数组维数

#解决切片引用会造成原数据改变，可以用copy()
#ndarray.copy()
#与切片不同，花式索引返回的是原对象的一个复制而不是引用。
#花式索引：通过构建好的列表作为索引；使用布尔数组索引；布尔表达式生成mask索引

#where语句
# where(array)
# where 函数会返回所有非零元素的索引。注意返回的是索引位置




