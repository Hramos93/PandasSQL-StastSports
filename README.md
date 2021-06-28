```python
import pandas as pd
from pandasql import sqldf 
import numpy as np
import matplotlib.pyplot as plt


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
newdf.Team= newdf.Team.str.replace('[\d+\W]','', regex=True)
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
count = pd.concat([countageM,countageF],keys=['Male','Female']).droplevel(level=1)
count.sort_values(by='Age', ascending=False)
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
      <th>Age</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Male</th>
      <td>71.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>70.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Female</th>
      <td>69.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>68.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>67.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Female</th>
      <td>13.0</td>
      <td>133</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>13.0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Female</th>
      <td>12.0</td>
      <td>24</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>12.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Female</th>
      <td>11.0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>114 rows × 2 columns</p>
</div>




```python
fig, ax = plt.subplots()
M = ax.bar(countageM['Age'],countageM['count'], label='Male', alpha=0.6)
F = ax.bar(countageF['Age'], countageF['count'], label='Female', alpha=0.6)
ax.legend()
plt.grid()
plt.show()
```


    
![png](output_10_0.png)
    



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
def stastplot(df:pd.DataFrame):
    
    assert type(df) == pd.DataFrame, "First parameter should be a DataFrame"
    
    x = np.arange(len(df.iloc[0].index))
    width = 0.2
    for i in range(0,len(summary),1):
        if i <= 0:
            plt.bar(x , df.iloc[i], width=0.2, label='Male',alpha=0.6)
        else:
            plt.bar(x + (width*1), df.iloc[i],width=0.2, label='Female',alpha=0.6)
    plt.xticks(x, ['min','avg','max'], horizontalalignment="left")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()
    
```


```python
stastplot(summary)
```


    
![png](output_14_0.png)
    



```python
Male.head()
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
      <th>ID</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Team</th>
      <th>Year</th>
      <th>Season</th>
      <th>City</th>
      <th>Sport</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>M</td>
      <td>24.0</td>
      <td>180.0</td>
      <td>80.0</td>
      <td>China</td>
      <td>1992</td>
      <td>Summer</td>
      <td>Barcelona</td>
      <td>Basketball</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>M</td>
      <td>23.0</td>
      <td>170.0</td>
      <td>60.0</td>
      <td>China</td>
      <td>2012</td>
      <td>Summer</td>
      <td>London</td>
      <td>Judo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>M</td>
      <td>31.0</td>
      <td>188.0</td>
      <td>75.0</td>
      <td>UnitedStates</td>
      <td>1992</td>
      <td>Winter</td>
      <td>Albertville</td>
      <td>Cross Country Skiing</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>M</td>
      <td>31.0</td>
      <td>188.0</td>
      <td>75.0</td>
      <td>UnitedStates</td>
      <td>1992</td>
      <td>Winter</td>
      <td>Albertville</td>
      <td>Cross Country Skiing</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>M</td>
      <td>31.0</td>
      <td>188.0</td>
      <td>75.0</td>
      <td>UnitedStates</td>
      <td>1992</td>
      <td>Winter</td>
      <td>Albertville</td>
      <td>Cross Country Skiing</td>
    </tr>
  </tbody>
</table>
</div>




```python
def team(bdd, groupby:str, orderby:str):
    qry = sql('''
    SELECT
    Team,
    Count({count}) as count
    FROM {df}
    Group by {group}
    Order by {order} desc
    '''.format(count = 'ID', df=bdd, group=groupby,order=orderby))
    return qry

```


```python
M = team('Male','Team','count')
F = team('Female','Team','count')
```


```python
def joinMF():
    qry = sql('''
    -- METRICS
    SELECT
    Team,
    Mcount,
    Fcount,
    (Mcount - Fcount) as diference,
    ROUND(CAST(Mcount as float) / CAST(Fcount as float),2)  as div
    FROM
    (-- FILL VALUE WITH 0 WHERE FCOUNT IS NAN
    SELECT
    Team,
    Mcount,
    COALESCE(Fcount,0) as Fcount
    FROM 
    ( -- JOIN TABLE M,F
    SELECT
    M.Team,
    M.count as Mcount,
    F.count as Fcount
    FROM M 
    LEFT JOIN F
    ON M.Team = F.Team))
    ''')
    return qry


```


```python
joinMF = joinMF()
joinMF = joinMF[~joinMF['div'].isnull()]
```


```python
joinMF50 = joinMF.head(50)
joinMF50 = joinMF50.sort_values(by='div', ascending=False)
```


```python
plt.figure(figsize=(12,6))
plt.bar(joinMF50['Team'],joinMF50['div'])
plt.xticks(rotation=45,horizontalalignment="right")
plt.title("Relation between quantity men and women by team")
plt.show()
```


    
![png](output_21_0.png)
    



```python
teamByYear = pd.DataFrame(Male.groupby(['Team','Year']).count()['ID'])
teamByYear.head()
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
      <th></th>
      <th>ID</th>
    </tr>
    <tr>
      <th>Team</th>
      <th>Year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acipactli</th>
      <th>1964</th>
      <td>3</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Afghanistan</th>
      <th>1960</th>
      <td>16</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>2</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>5</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
def teambyyear(df:str):    
    qry = sql('''
    SELECT
    Team,
    Year,
    Count('ID') as count{0}
    FROM {0}
    Group by Team,Year
    '''.format(df))
    return qry

```


```python
M = teambyyear('Male')
F = teambyyear('Female')
```


```python
testM = M[M['Team'] == 'Iran'] 
testF = F[F['Team'] == 'Iran']

```


```python
def country(country:str, team='Team'):
    Male =M[M[team] ==country]
    Fem = F[F[team] == country]
    merge = Male.merge(Fem, on='Year', how='left')
    merge = merge.fillna(0)
    merge = merge.drop(['Team_y'],axis=1)
    merge.columns = ['Country','Year','Male','Female']
    return merge
```


```python
def teambycountry(country):
    qry=sql('''
    SELECT
    Team,
    Year,
    Male,
    COALESCE(Female,0) as Female
    FROM (
    -- SELECT COLUMNS
    SELECT
    M.Team,
    M.Year,
    M.countMale as Male,
    F.countFemale as Female
    FROM
    -- JOIN M with F
    (SELECT
    *
    FROM M
    WHERE Team = "{0}") as M
    LEFT  JOIN (
    -- DATASET FEMALE
    SELECT
    *
    FROM F
    WHERE Team ="{0}") as F
    ON M.Year = F.Year)
    '''.format(country))
    return qry
```


```python
def barplot(df):
    country = teambycountry(df)
    plt.figure(figsize=(12,6))
    width = 2
    plt.bar(country.Year, country['Male'], width=width)
    plt.bar(country.Year, country['Female'],width=width)
    plt.title('{}'.format(df))
    plt.show
```


```python
barplot('Venezuela')
```


    
![png](output_29_0.png)
    



```python

```
