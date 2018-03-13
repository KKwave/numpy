import  numpy as np
import  matplotlib.pyplot as plt
import  math
import  pandas as pd
import math

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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

#查看所用字节
#ndarray.nbytes

#转换数组类型，使用asarray
#asarray(ndarray,dtype = float64)
#astype()则会返回一个新数组,总是返回原来数组的一份复制，即使转换的类型是相同的
#ndarray.astype(float64)

# %precision 3 表示数据的小数精度在3位

#最大最小值的位置
#使用 argmin, argmax 方法

#clip方法，将数值限制在某个范围
#如ndarray.clip(3,5)  小于3的变成3，大于5的变成5

#ptp 方法
#ndarray.ptp() 计算最大值和最小值之差

#np.sort()  返回的结果是从小到大排列的 调用sort方法会改变数组的值，传入axis会改变排序方式
#np.argsort()返回从小到大的排列在数组中的索引位置

#searchsorted 函数
#searchsorted 接受两个参数，其中，第一个必需是已排序的数组
#sorted_array = linspace(0,1,5)
#values = array([.1,.8,.3,.12,.5,.25])
#np.searchsorted(sorted_array, values)

#使用 newaxis 增加数组维数
#a = arange(3)
#y = a[newaxis, :]

#squeeze 方法去除多余的轴
#squeeze 返回一个将所有长度为1的维度去除的新数组。

#需要注意使用转置时 a.T， 对转置修改会改变原数组

#atleast_xd 函数
#保证数组至少有 x 维：
#atleast_1d(x)保证1维 atleast_2d(x)保证2维

#查看数组对角线元素：
#a.diagonal()
#可以使用偏移来查看它的次对角线，正数表示右移，负数表示左移：
#a.diagonal(offset=1)

#数组与字符串的转换
#ndarray.tostring()
#fromstring 函数 可以使用 fromstring 函数从字符串中读出数据，不过要指定类型
#s = a.tostring()
#a = np.fromstring(s,dtype=np.uint8)
#此时，返回的数组是一维的，需要重新设定维度,如reshape
# 对于文本文件，推荐使用
# loadtxt
# genfromtxt
# savetxt
# 对于二进制文本文件，推荐使用
# save
# load
# savez

#tolist方法转换为列表
#ndarray.tolist()

#保存
#保存成文本 a.dump("file.txt")
#字符串：a.dumps()
#写入文件：a.tofile('foo.csv', sep=',', format="%s")

#meshgrid 有时候需要在二维平面中生成一个网格，这时候可以使用 meshgrid 来完成这样的工作
# x_ticks = np.linspace(-1, 1, 5)
# y_ticks = np.linspace(-1, 1, 5)
# x, y = np.meshgrid(x_ticks, y_ticks)

#identity产生一个 n 乘 n 的单位矩阵
#np.identity(3) 就会产生一个3*3的单位矩阵

#ndarray.I表示矩阵的逆矩阵

#hypot 返回对应点 (x,y) 到原点的距离。
# x = np.array([1,2,3])
# y = np.array([4,5,6])
# np.hypot(x,y)

#np.nonzero()  选取非零的索引位置
# a = np.array([0,1,2,0,4,0])
# print(np.nonzero(a))
# b = a[np.nonzero(a)]
# print(b)

#向量化函数
#自定义的 sinc 函数：
# def sinc(x):
#     if x == 0.0:
#         return 1.0
#     else:
#         w = np.pi * x
#         return np.sin(w) / w
#只能作用于单个数值，不能作用于数组
#可以使用 numpy 的 vectorize 将函数 sinc 向量化，产生一个新的函数：
#vsinc = np.vectorize(sinc)
#vsinc(x) 其作用是为 x 中的每一个值调用 sinc 函数
#因为这样的用法涉及大量的函数调用，因此，向量化函数的效率并不高

#四则运算的二元方式
#运算	        函数
# a + b   	add(a,b)
# a - b	   subtract(a,b)
# a * b	   multiply(a,b)
# a / b	    divide(a,b)
# a ** b	power(a,b)
# a % b  	remainder(a,b)
#函数还可以接受第三个参数，表示将结果存入第三个参数中

#对于两个数组进行比较
#直接a == b会判断对应的每个数
#如果我们在条件中要判断两个数组是否一样要使用
# all(a==b)
#对于浮点数，由于存在精度问题，使用函数 allclose 会更好：
# if allclose(a,b)
#logical_and 也是逐元素的 and 操作：
#np.logical_and(a, b)
#0 被认为是 False，非零则是 True

#numpy也有reduce方法
#accumulate 方法可以看成保存 reduce 每一步的结果所形成的数组

#reduceat 方法将操作符运用到指定的下标上，返回一个与 indices 大小相同的数组
# a = np.array([0, 10, 20, 30, 40, 50])
# indices = np.array([1,4])
# np.add.reduceat(a, indices)
#输出array([60, 90])
#这里，indices 为 [1, 4]，所以 60 表示从下标1（包括）加到下标4（不包括）的结果，90 表示从下标4（包括）加到结尾的结果

#outer 方法对于 a 中每个元素，将 op 运用到它和 b 的每一个元素上所得到的结果
# a = np.array([0,1])
# b = np.array([1,2,3])
# np.add.outer(a, b) #会把a每个元素加上每个b的元素
#注意有顺序的区别：
#np.add.outer(b, a)


#choose函数
#进行条件选择，此时使用 choose 函数十分方便
#可以实现将数组中所有小于 10 的值变成了 10，大于 15 的值变成了 15
#即choose(a.b)取a的架构，用b的item填写

#loadtxt 函数
# loadtxt(fname, dtype=<type 'float'>,
#         comments='#', delimiter=None,
#         converters=None, skiprows=0,
#         usecols=None, unpack=False, ndmin=0)
#loadtxt 有很多可选参数，其中 delimiter 就是分隔符参数。
#skiprows 参数表示忽略开头的行数，可以用来读写含有标题的文本
#data = np.loadtxt('myfile.txt',
                 #  skiprows=1,         #忽略第一行
                 #  dtype=np.int,      #数组类型 dtype=np.object, #数据类型为对象
                 #  delimiter=',',     #逗号分割
                 #  usecols=(0,1,2,4), #指定使用哪几列数据
                 #  comments='%'       #百分号为注释符
                 #  converters={0:date_converter,  #第一列使用自定义转换方法
                 #              1:float,           #第二第三使用浮点数转换
                 #              2:float})
                 # )
# 移除 myfile.txt：
# import os
# os.remove('myfile.txt')

# 读写各种格式的文件
# 如下表所示：
#
# 文件格式	              使用的包	                                  函数
# txt	                     numpy                              	loadtxt, genfromtxt, fromfile, savetxt, tofile
# csv	                        csv                                 reader, writer
# Matlab	                scipy.io	                            loadmat, savemat
# hdf	                   pytables, h5py
# NetCDF	        netCDF4, scipy.io.netcdf	                     netCDF4.Dataset, scipy.io.netcdf.netcdf_file
# 文件格式	             使用的包                                 	备注
# wav	             scipy.io.wavfile	                           音频文件
# jpeg,png,...     	PIL, scipy.misc.pilutil	                        图像文件
# fits	                  pyfits	                               天文图像

#将数组写入文件
#savetxt 可以将数组写入文件，默认使用科学计数法的形式保存
#np.savetxt('out.txt', data，fmt="%d"#保存为整数)

#结构化数据
#定义一个人的结构类型：
#person_dtype = np.dtype([('name', 'S10'), ('age', 'int'), ('weight', 'float')])
#查看类型所占字节数：
#person_dtype.itemsize
# 产生一个 3 x 4 共12人的空结构体数组：
# people = np.empty((3,4), person_dtype)
# 分别赋值：
# people['name'] = [['Brad', 'Jane', 'John', 'Fred'],
#                   ['Henry', 'George', 'Brain', 'Amy'],
#                   ['Ron', 'Susan', 'Jennife', 'Jill']]
# people['age'] = [[33, 25, 47, 54],
#                  [29, 61, 32, 27],
#                  [19, 33, 18, 54]]
# people['weight'] = [[135., 105., 255., 140.],
#                     [154., 202., 137., 187.],
#                     [188., 135., 88., 145.]]
# print people

#读取结构化数组
# 利用 loadtxt 指定数据类型，从这个文件中读取结构化数组：
# person_dtype = np.dtype([('name', 'S10'), ('age', 'int'), ('weight', 'float')])
# people = np.loadtxt('people.txt',
#                     skiprows=1,
#                     dtype=person_dtype)
