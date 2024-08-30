#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
student={'Unit Test-1':[5,6,8,3,10],'Unit Test-2':[7,8,9,6,15]}
student1={'Unit Test-1':[3,3,6,6,8],'Unit Test-2':[5,9,8,6,5]}
ds=pd.DataFrame(student)
ds1=pd.DataFrame(student1)
print(ds)
print(ds1)
print("Subtraction")
print(ds.sub(ds1))
print("rsub")
print(ds.sub(ds1))
print("Addition")
print(ds.add(ds1))
print("radd")
print(ds.radd(ds1))
print("Multiplication")
print(ds.mul(ds1))
print("Division")
print(ds.div(ds1))


# In[6]:


import pandas as pd
d1={'roll_no':[10,11,12,13,14,15],'name':['Ankit','Pihu','Rinku','Yash','Vijay','Nikhil']}
df1=pd.DataFrame(d1,columns=['roll_no','name'])
print(df1)
d2={'roll_no':[1,2,3,4,5,6],'name':['Renu','Jatin','Deep','Guddu','Chhaya','Sahil']}
df2=pd.DataFrame(d1,columns=['roll_no','name'])
print(df2)


# In[12]:


import pandas as pd
grade={'Name':['Rashmi','Harsh','Ganesh','Priya','Vivek','Anita','Karthik'],'Grade':['A1','A2','B1','A1','B2','A2','A1']}
gr=pd.DataFrame(grade,columns=['Name','Grade'])
print(gr)
print(gr.iloc[0:5])
print(gr[0:5])
gr['Percentage']=[92,89,None,95,68,None,93]
print(gr)
gr=gr.reindex(['Name','Percentage','Grade'],axis=1)#reaarnges column
print(gr)
print(gr.iloc[:,[0,2,1]])#rearanges column
gr=gr.drop('Grade',axis=1)#drops column,axis=1 for columns
print(gr)
gr=gr.drop([2,4],axis=0)#axis=0 for rows
print(gr)


# In[ ]:




