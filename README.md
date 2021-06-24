```python
import pandas as pd
from pandasql import sqldf 
import numpy as np
import matplotlib.pyplot as plt

import re
import traceback
```


```python
## Varname
def varname(var):
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    return vars_name
```


```python
sql = lambda q: sqldf(q, globals())
df = pd.read_csv("athlete_events.csv")
df.columns
```




    Index(['ID', 'Name', 'Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC', 'Games',
           'Year', 'Season', 'City', 'Sport', 'Event', 'Medal'],
          dtype='object')




```python
def fun(bbd):
    qry =sql('''
    SELECT
    *
    FROM  df
    LIMIT 5
    '''.format(df = varname(bbd)))
    return qry
```


```python
def clean(bbd:str,columns:list):
    qry = sql('''
    SELECT
    {1}
    FROM {0}
    '''.format(bbd,str(columns)[1:-1].replace("'" , "")))
    return qry
```


```python
columns = 'ID','Sex','Age','Height','Weight','Team','Year','Season','City','Sport'
newdf = clean('df',columns)
newdf = newdf.dropna(subset=['Age','Height','Weight'])
```


```python
def sex(bbd, columns:list, sex:str):
    qry = sql('''
    SELECT
    {1}
    FROM {0}
    WHERE Sex = "{2}"
    '''.format(bbd,str(columns)[1:-1].replace("'" , ""), sex))
    return qry
```


```python
Male = sex('newdf',columns,'M')
Female = sex('newdf', columns, 'F')
print(' Shape Male DataFrame {} \n Shape Female DataFrame {}'.format(Male.shape, Female.shape))
```

     Shape Male DataFrame (139454, 10) 
     Shape Female DataFrame (66711, 10)
    

### Male


```python
def Count(bbd:str, column:str):
    qry= sql('''
    SELECT
    {1},
    COUNT({1}) as count
    FROM {0}
    GROUP BY {1}
    '''.format(bbd,column))
    return qry
```


```python
countageM = Count('Male','Age')
countageF = Count('Female','Age')
```


```python
fig, ax = plt.subplots()
Male = ax.bar(countageM['Age'],countageM['count'], label='Male', alpha=0.6)
Female = ax.bar(countageF['Age'], countageF['count'], label='Female', alpha=0.6)
ax.legend()
plt.grid()
plt.show()
```


    
![png](output_11_0.png)
    



```python
def stast(bbd:str, column, sex:str):
    qry=sql('''
    SELECT
    min({1}) as min,
    ROUND(avg({1}),2) as avg,
    max({1}) as max
    FROM {0}
    '''.format(bbd, column,sex))
    return qry
```


```python
stastM = stast('countageM','Age','Male')
stastF = stast('countageF', 'Age','Female')
summary = pd.concat([stastM,stastF],keys=['Male','Female']).droplevel(level=1)
```


```python
def stastplot(df):
    x = np.arange(len(df.iloc[0].index))
    width = 0.2
    plt.bar(x, df.iloc[0], width=0.2, label='Male',alpha=0.6)
    plt.bar(x + width, df.iloc[1],width=0.2, label='Female',alpha=0.6)
    plt.xticks(x, ['min','avg','max'], horizontalalignment="left")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()
    
```


```python
stastplot(summary)
```


    
![png](output_15_0.png)
    



```python
summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>avg</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Male</th>
      <td>12.0</td>
      <td>41.03</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>Female</th>
      <td>11.0</td>
      <td>38.22</td>
      <td>69.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
