#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
arr=np.array([1,2,3,4])
print(arr)


# In[5]:


import numpy as np
arr=np.array([[1,2,3,4],[5,6,7,8]])
print(arr[1,-1])
print(arr.shape)


# In[6]:


import numpy as np
arr=np.array([[1,2,3,4],[5,6,7,8]])
for x in arr:
    for y in x:
        print(y)


# In[7]:


import numpy as np
arr=np.array([1,2,3,4])
for x in arr:
    print(x)


# In[9]:


import numpy as np
arr1=np.array([1,2,3,4])
arr2=np.array([5,6,7,8])
arr=np.concatenate((arr1,arr2),axis=0)
print(arr)


# In[13]:


import numpy as np
arr1=np.array([[1,2,3,4],[6,3,2,2]])
arr2=np.array([[5,6,7,8],[3,9,2,1]])
arr=np.concatenate((arr1,arr2),axis=0)
print(arr)


# In[12]:


import numpy as np
arr=np.array([9,2,3,1])
arr1=np.sort(arr)
print(arr1)


# In[ ]:





# In[17]:


from numpy import random
x=random.randint(100)
print(x)


# In[18]:


from numpy import random
x=random.rand()
print(x)


# In[19]:


from numpy import random
x=random.rand(3,5)
print(x)


# In[21]:


from numpy import random
x=random.randint(100,size=(3,5))
print(x)


# In[22]:


from numpy import random
x=random.rand(5)
print(x)


# In[ ]:





# In[30]:


import numpy as np
arr1=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
arr=arr1.reshape(3,4)
print(arr)


# In[ ]:




