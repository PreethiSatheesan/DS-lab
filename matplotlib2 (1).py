#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
t=np.arange(0.0,20.0,1)
s=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
s2=[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
plt.subplot(2,1,1)
plt.ylabel('Value')
plt.title('First chart')
plt.grid(True)
plt.plot(t,s)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.subplot(2,1,2)
plt.xlabel('Item(s)')
plt.ylabel('Value')
plt.title('Second chart')
plt.plot(t,s2)
plt.grid(True)
plt.show()


# In[6]:


for x in range(11):
    print(x)


# In[7]:


for x in range(2,11):
    print(x)


# In[8]:


for x in range(1,11,2):
    print(x)


# In[9]:


import numpy as np
x=np.arange(1,10,3)
print(x)


# In[11]:


import numpy as np
x=np.arange(10)
print(x)


# In[12]:


import numpy as np
x=np.arange(1,10)
print(x)


# In[13]:


import matplotlib.pyplot as plt
y_axis=[20,50,30]
x_axis=range(len(y_axis))
plt.bar(x_axis,y_axis,width=.5,color='orange')
plt.show()


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
objects=('python','c++','java','perl','scala','lisp')
y_pos=np.arange(len(objects))
performance=[10,8,6,4,2,1]
plt.barh(y_pos,performance,align='center',color='r')
plt.yticks(y_pos,objects)
plt.xlabel('usage')
plt.title('programming language usage')
plt.show()


# In[ ]:




