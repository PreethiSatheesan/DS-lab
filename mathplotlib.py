#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
plt.plot([1,2,3],[5,7,4])
plt.show()


# In[2]:


import matplotlib.pyplot as plt
x=[1,2,3]
y=[5,7,4]
plt.plot(x,y,label='First line')
x2=[1,2,3]
y2=[10,11,14]
plt.plot(x2,y2,label='second line')
plt.xlabel('Plot number')
plt.ylabel('import variables')
plt.title('new graph')
plt.legend()
plt.show()


# In[ ]:




