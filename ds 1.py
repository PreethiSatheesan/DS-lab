#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
arr=np.array([[1,2,3,1],[4,5,6,2],[7,8,9,3]])
print(arr)
print(arr[:2,:3])
print(arr[:3,::2])
print(arr[::-1,::-1])
print(arr[1:2:1])
print(arr[:,0])
print(arr[0,:])
print(arr[0])


# In[38]:


import numpy as np
a=np.array([1,2,3])
print("Type:%s"%type(a))
print("Shape:%s"%a.shape)
print(a[0],a[1],a[2])

a[0]=5
print(a)

b=np.array([[1,2,3],[4,5,6]])
print("\n Shape of b:",b.shape)
print(b[0,0],b[0,1],b[1,0])

a=np.zeros((2,2))
print("All zero matrix:\n %s"%a)

b=np.ones((1,2))
print("\n All ones matrix:\n %s"%b)

d=np.eye(2)
print("\nIdentity matrix:\n %s"%d)

e=np.random.random((2,2))
print("\nRandom matrix:\n %s"%e)

print("Vectorized sum example \n")
x=np.array([[1,2],[3,4]])
print("X:\n%s"%x)
print("Sum:%s"%np.sum(x))
print("Summ axis=0:%s"%np.sum(x,axis=0))
print("Summ axis=1:%s"%np.sum(x,axis=1))

a=np.arange(10000)
b=np.arange(10000)
dp=np.dot(a,b)
print("Dot Product:%s \n"%dp)
op=np.outer(a,b)
print("\nOuter Porduct:%s \n"%op)
ep=np.multiply(a,b)
print("\nElement wise product:%s \n"%ep)


# In[41]:


#matrix transformation
import numpy as np
x=np.array([[1,2],[3,4]])
print("Original x:\n %s"%x)
print("\nTranspose of x:\n%s"%x.T)


# In[42]:


#SVD using python
from numpy import array
from scipy.linalg import svd
A=array([[1,2],[3,4],[5,6]])
print("A:\n %s"%A)
U,S,VT=svd(A)
print("\nU:\n%s"%U)
print("\nS:\n%s"%S)
print("\nV^T:\n%s"%VT)


# In[ ]:




