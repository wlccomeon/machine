# -*- coding: utf-8 -*-

import sys
from datetime import datetime
import numpy as np

print "\n"
print "**********向量相加**********"
#向量相加-Python
def pythonsum(n):
    a = range(n)
    b = range(n)
    c = []
    for i in range(len(a)):
        a[i] = i ** 2
        b[i] = i ** 3
        c.append(a[i] + b[i])
    return c

#向量相加-NumPy
def numpysum(n):
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3
    c = a + b
    return c

#效率比较
size = 1000
start = datetime.now()
c = pythonsum(size)
delta = datetime.now() - start
print "The last 2 elements of the sum:\n", c[-2:]
print "PythonSum elapsed time in microseconds:\n", delta.microseconds

start = datetime.now()
c = numpysum(size)
delta = datetime.now() - start
print "The last 2 elements of the sum:\n", c[-2:]
print "NumPySum elapsed time in microseconds:\n", delta.microseconds

print "\n"
print "**********numpy数组**********"
#numpy数组
a = np.arange(5)
print "np.arange(5):\n",a
print "a.dtype:\n",a.dtype
print "a.shape:\n",a.shape

a = range(5)
print "a = range(5):\n",a
print "type(a):\n",type(a)
#print "a.dtype",a.dtype
#print "a.shape",a.shape

print "\n"
print "**********创建多维数组**********"
#创建多维数组
m = np.array([np.arange(2), np.arange(2)])
print "创建多维数组m:\n",m
print "多维数组m.shape:\n",m.shape

print "\n"
print "**********数组内元素的类型**********"
#数组内元素的类型
print "多维数组m.dtype:\n",m.dtype
print "np.zeros(10):\n",np.zeros(10)
print "np.zeros((3, 6)):\n",np.zeros((3, 6))
print "np.empty((2, 3, 2)):\n",np.empty((2, 3, 2))
print "np.arange(15):\n",np.arange(15)
print "type(np.arange(15)):\n",type(np.arange(15))
print "range(15):\n",range(15)
print "type(range(15)):\n",type(range(15))

print "\n"
print "**********选取数组元素**********"
#选取数组元素
a = np.array([[1,2],[3,4]])
print "In: a:",a
print "In: a[0,0]:\n", a[0,0]
print "In: a[0,1]:\n", a[0,1]
print "In: a[1,0]:\n",a[1,0]
print "In: a[1,1]:\n",a[1,1]

print "\n"
print "**********numpy数据类型**********"
#numpy数据类型
print "In: float64(42):\n", np.float64(42)
print "In: int8(42.0):\n",np.int8(42.0)
print "In: bool(42):\n",np.bool(42)
print "np.bool(0):\n",np.bool(0)
print "In: bool(42.0):\n", np.bool(42.0)
print "In: float(True):\n",np.float(True)
print "np.float(False):\n",np.float(False)
print "In: arange(7, dtype=uint16):\n",np.arange(7, dtype=np.uint16)

#抓取错误
print "错误抓取处理 In: int(42.0 + 1.j):"
try:
   print np.int(42.0 + 1.j)
except TypeError:
   print "TypeError"

#不用错误抓取处理 type error
#print float(42.0 + 1.j)

print "\n"
print "**********数据类型转换**********"
# 数据类型转换
arr = np.array([1, 2, 3, 4, 5])
print "arr:",arr
print "arr.dtype:",arr.dtype
float_arr = arr.astype(np.float64)
print "float_arr:",float_arr
print "float_arr.dtype",float_arr.dtype

arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
print "arr:",arr
print "arr.astype(np.int32):",arr.astype(np.int32)

numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
print "numeric_strings:",numeric_strings
print "numeric_strings.astype(float):",numeric_strings.astype(float)


print "\n"
print "**********数据类型对象**********"
#数据类型对象
a = np.array([[1,2],[3,4]])
print "np.array([[1,2],[3,4]]):",np.array([[1,2],[3,4]])
#TODO：byteorder 
print "a.dtype.byteorder:",a.dtype.byteorder
print "a.dtype.itemsize:",a.dtype.itemsize

print "\n"
print "**********字符编码**********"
#字符编码
print "np.arange(7, dtype='f'):",np.arange(7, dtype='f')
print "np.arange(7, dtype='D'):",np.arange(7, dtype='D')
print "np.dtype(float):",np.dtype(float)
print "np.dtype('f'):",np.dtype('f')
print "np.dtype('d'):",np.dtype('d')
print "np.dtype('f8'):",np.dtype('f8')
print "np.dtype('Float64'):",np.dtype('Float64')

print "\n"
print "**********dtype类的属性**********"
#dtype类的属性
t = np.dtype('Float64')
print "np.dtype('Float64'):",np.dtype('Float64')
print "t.char:",t.char
print "t.type:",t.type
print "t.str:",t.str

print "\n"
print "**********创建自定义数据类型**********"
#创建自定义数据类型
t = np.dtype([('name', np.str_, 40), ('numitems', np.int32), ('price', np.float32)])
print "t:",t
print "t['name']:",t['name']

itemz = np.array([('Meaning of life DVD', 42, 3.14), ('Butter', 13, 2.72)], dtype=t)
print "itemz:",itemz
print "itemz[1]:",itemz[1]

print "\n"
print "**********数组与标量的运算**********"
#数组与标量的运算
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
print "arr:",arr
print "arr * arr:",arr * arr
print "arr - arr:",arr - arr
print "1 / arr:",1 / arr
print "arr ** 0.5",arr ** 0.5

print "\n"
print "**********一维数组的索引与切片**********"
#一维数组的索引与切片
a = np.arange(9)
print "a:",a
print "a[3:7]:",a[3:7]
print "a[:7:2]:",a[:7:2]
print "a[::-1]:",a[::-1]

s = slice(3,7,2)
print "s:",s
print "a[s]:",a[s]

s = slice(None, None, -1)
print "s:",s
print "a[s]",a[s]

print "\n"
print "**********多维数组的切片与索引**********"
#多维数组的切片与索引
#TODO:多维数组不理解？？reshape（）的含义？？
b = np.arange(24).reshape(2,3,4)

print "b.shape:",b.shape
print "b:",b
print "b[0,0,0]:",b[0,0,0]
print "b[:,0,0]:",b[:,0,0]
print "b[0]:",b[0]
print "b[0, :, :]:",b[0, :, :]
print "b[0, ...]:",b[0, ...]
print "b[0,1]",b[0,1]
print "b[0,1,::2]:",b[0,1,::2]
print "b[...,1]:",b[...,1]
print "b[:,1]:",b[:,1]
print "b[0,:,1]:",b[0,:,1]
print "b[0,:,-1]:",b[0,:,-1]
print "b[0,::-1, -1]:",b[0,::-1, -1]
print "b[0,::2,-1]",b[0,::2,-1]
print "b[::-1]",b[::-1]

s = slice(None, None, -1)
print b[(s, s, s)]

print "\n"
print "**********布尔型索引**********"
#布尔型索引
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
#TODO:不懂生成这样一个矩阵的意义何在？？ (在Matlab中，rand是0-1的均匀随机分布，而randn是均值为0方差为1的正态分布。) add by why
data = np.random.randn(7, 4)
print "names:",names
print "data",data

print "names == 'Bob':",names == 'Bob'
print "data[names == 'Bob']:",data[names == 'Bob']
#TODO:下面两行中的2:，3 不理解
print "data[names == 'Bob', 2:]:",data[names == 'Bob', 2:]
print "data[names == 'Bob', 3]:",data[names == 'Bob', 3]
print "names != 'Bob':",names != 'Bob'

#TypeError: The numpy boolean negative, the `-` operator, is not supported, use the `~` operator or the logical_not function instead.
#data[-(names == 'Bob')]
print "data[~(names == 'Bob')]:",data[~(names == 'Bob')]
mask = (names == 'Bob') | (names == 'Will')
print "(names == 'Bob') | (names == 'Will'):",mask
print "data[mask]:",data[mask]
data[data < 0] = 0
print "data[data < 0] = 0：",data
data[names != 'Joe'] = 7
print "data[names != 'Joe'] = 7：",data


print "\n"
print "**********花式索引**********"
#花式索引
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
print "arr:",arr
print "arr[:]:",arr[:]
print "arr[1]:",arr[1]
print "arr[:1]:",arr[:1]
#TODO:下面两种arr[[4, 3, 0, 6]]；arr[[-3, -5, -7]] 理解不了 ？？
print "arr[[4, 3, 0, 6]]:",arr[[4, 3, 0, 6]]
print "arr[[-3, -5, -7]]:",arr[[-3, -5, -7]]
arr = np.arange(32).reshape((8, 4))
print "np.arange(32).reshape((8, 4)):",arr
#TODO:下面两种arr[[1, 5, 7, 2], [0, 3, 1, 2]]；arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])] 理解不了 ？？
print "arr[[1, 5, 7, 2], [0, 3, 1, 2]]:",arr[[1, 5, 7, 2], [0, 3, 1, 2]]
print "arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]:",arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]

print "\n"
print "**********数组转置**********"
#数组转置
arr = np.arange(15).reshape((3, 5))
print "np.arange(15).reshape((3, 5)):",arr
print "arr.T:",arr.T

print "\n"
print "**********改变数组的维度**********"
#改变数组的维度
b = np.arange(24).reshape(2,3,4)
print "np.arange(24).reshape(2,3,4):\n",b
print "b.ravel():\n",b.ravel()
print "b.flatten():\n",b.flatten()
b.shape = (6,4)
print "b.shape = (6,4):\n",b.shape
print "b:\n",b
print "b.transpose():\n",b.transpose()
b.resize((2,12))
print "b.resize((2,12))：\n",b


print "\n"
print "**********组合数组**********"
#组合数组
a = np.arange(9).reshape(3,3)
print "a = np.arange(9).reshape(3,3):\n",a
b = 2 * a
print "b = 2 * a:\n",b
print "np.hstack((a, b)):\n",np.hstack((a, b))
print "np.concatenate((a, b), axis=1):\n",np.concatenate((a, b), axis=1)
print "np.vstack((a, b)):\n",np.vstack((a, b))
print "np.concatenate((a, b), axis=0):\n",np.concatenate((a, b), axis=0)
print "np.dstack((a, b)):\n",np.dstack((a, b))
oned = np.arange(2)
print "oned = np.arange(2)：\n",oned
twice_oned = 2 * oned
print "twice_oned = 2 * oned:\n",twice_oned
print "np.column_stack((oned, twice_oned)):\n",np.column_stack((oned, twice_oned))
print "np.column_stack((a, b)):\n",np.column_stack((a, b))
print "np.column_stack((a, b)) == np.hstack((a, b)):\n",np.column_stack((a, b)) == np.hstack((a, b))
print "np.row_stack((oned, twice_oned)):\n",np.row_stack((oned, twice_oned))
print "np.row_stack((a, b)):\n",np.row_stack((a, b))
print "np.row_stack((a,b)) == np.vstack((a, b)):\n",np.row_stack((a,b)) == np.vstack((a, b))


print "\n"
print "**********数组的分割**********"
#数组的分割
a = np.arange(9).reshape(3, 3)
print "a = np.arange(9).reshape(3, 3):\n",a
print "\n"
print "竖向分割"
print "np.hsplit(a, 3):\n",np.hsplit(a, 3)
print "np.split(a, 3, axis=1):\n",np.split(a, 3, axis=1)
print "\n"
print "横向分割"
print "np.vsplit(a, 3):\n",np.vsplit(a, 3)
print "np.split(a, 3, axis=0):\n",np.split(a, 3, axis=0)
print "\n"
c = np.arange(27).reshape(3, 3, 3)
print "c = np.arange(27).reshape(3, 3, 3):\n",c
print "np.dsplit(c, 3):\n",np.dsplit(c, 3)


print "\n"
print "**********数组的属性**********"
#数组的属性
b=np.arange(24).reshape(2,12)
print "b=np.arange(24).reshape(2,12):\n",b
print "b.ndim:\n",b.ndim
print "b.size:\n",b.size
print "b.itemsize:\n",b.itemsize
print "b.nbytes:\n",b.nbytes

print "\n"
b = np.array([ 1.+1.j,  3.+2.j])
print "b = np.array([ 1.+1.j,  3.+2.j]):\n",np.array([ 1.+1.j,  3.+2.j])
print "提取复数的实部b.real():\n",b.real
print "提取复数的虚部b.imag():\n",b.imag

print "\n"
b=np.arange(4).reshape(2,2)
print "b=np.arange(4).reshape(2,2)：\n",b
print "b.flat:\n",b.flat
print "b.flat[2]:\n",b.flat[2]


print "\n"
print "**********数组的转换**********"
#数组的转换
b = np.array([ 1.+1.j,  3.+2.j])
print "b = np.array([ 1.+1.j,  3.+2.j]):\n",b
print "b.tolist():\n",b.tolist()
print "b.tostring():\n",b.tostring()
print np.fromstring('\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x00@', dtype=complex)
print "np.fromstring('20:42:52',sep=':', dtype=int):\n",np.fromstring('20:42:52',sep=':', dtype=int)
print "b:\n",b
#print "b.astype(int):\n",b.astype(int)
print "b.astype('complex'):\n",b.astype('complex')
