```python
from sklearn.preprocessing import LabelEncoder
```


```python
items = ['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
```


```python
labels
```




    array([0, 1, 4, 5, 3, 3, 2, 2])




```python
encoder.transform(['냉장고'])
```




    array([1])




```python
encoder.classes_
```




    array(['TV', '냉장고', '믹서', '선풍기', '전자레인지', '컴퓨터'], dtype='<U5')




```python
encoder.inverse_transform([1])
```




    array(['냉장고'], dtype='<U5')




```python
encoder.inverse_transform(labels)
```




    array(['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서'], dtype='<U5')




```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np
```


```python
items

items_t = np.array(items).reshape(-1, 1)
items_t
```




    array([['TV'],
           ['냉장고'],
           ['전자레인지'],
           ['컴퓨터'],
           ['선풍기'],
           ['선풍기'],
           ['믹서'],
           ['믹서']], dtype='<U5')




```python
items_l = [['TV'],
       ['냉장고'],
       ['전자레인지'],
       ['컴퓨터'],
       ['선풍기'],
       ['선풍기'],
       ['믹서'],
       ['믹서']]
```


```python
oh_encoder = OneHotEncoder()
oh_encoder.fit(items_l)
result = oh_encoder.transform(items_l)
result.toarray()
```




    array([[1., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0., 0.],
           [0., 0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0.]])




```python
oh_encoder.categories_
```




    [array(['TV', '냉장고', '믹서', '선풍기', '전자레인지', '컴퓨터'], dtype=object)]




```python
oh_encoder.inverse_transform([[1., 0., 0., 0., 0., 0.]])
```




    array([['TV']], dtype=object)




```python
import pandas as pd
```


```python
df = pd.DataFrame({'item':['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']})
```


```python
pd.get_dummies(df['item'])
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
      <th>TV</th>
      <th>냉장고</th>
      <th>믹서</th>
      <th>선풍기</th>
      <th>전자레인지</th>
      <th>컴퓨터</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.datasets import load_iris
```


```python
iris = load_iris(as_frame=True)
```


```python
iris.data.mean()
```




    sepal length (cm)    5.843333
    sepal width (cm)     3.057333
    petal length (cm)    3.758000
    petal width (cm)     1.199333
    dtype: float64




```python
iris.data.var()
```




    sepal length (cm)    0.685694
    sepal width (cm)     0.189979
    petal length (cm)    3.116278
    petal width (cm)     0.581006
    dtype: float64




```python
from sklearn.preprocessing import StandardScaler
```


```python
scaler = StandardScaler()
scaler.fit(iris.data)
iris_scaled = scaler.transform(iris.data)
```


```python
iris_scaled.mean()
```




    -4.855375361027351e-16




```python
iris_df = pd.DataFrame(iris_scaled)
```


```python
iris_df.mean().round(5)
```




    0   -0.0
    1   -0.0
    2   -0.0
    3   -0.0
    dtype: float64




```python
iris_df.var()
```




    0    1.006711
    1    1.006711
    2    1.006711
    3    1.006711
    dtype: float64




```python
from sklearn.preprocessing import MinMaxScaler
```


```python
scaler = MinMaxScaler()
scaler.fit(iris.data)
iris_scaled = scaler.transform(iris.data)
```


```python
iris_df = pd.DataFrame(iris_scaled)
iris_df.min(),iris.data.min()
```




    (0    0.0
     1    0.0
     2    0.0
     3    0.0
     dtype: float64,
     sepal length (cm)    4.3
     sepal width (cm)     2.0
     petal length (cm)    1.0
     petal width (cm)     0.1
     dtype: float64)




```python
iris_df.max(), iris.data.max()
```




    (0    1.0
     1    1.0
     2    1.0
     3    1.0
     dtype: float64,
     sepal length (cm)    7.9
     sepal width (cm)     4.4
     petal length (cm)    6.9
     petal width (cm)     2.5
     dtype: float64)




```python
train_array = np.arange(0,11).reshape(-1,1)
train_array
```




    array([[ 0],
           [ 1],
           [ 2],
           [ 3],
           [ 4],
           [ 5],
           [ 6],
           [ 7],
           [ 8],
           [ 9],
           [10]])




```python
test_array = np.arange(0,6).reshape(-1,1)
test_array
```




    array([[0],
           [1],
           [2],
           [3],
           [4],
           [5]])




```python
scaler = MinMaxScaler()
scaler.fit(train_array)
train_array = scaler.transform(train_array)
```


```python
train_array
```




    array([[0. ],
           [0.1],
           [0.2],
           [0.3],
           [0.4],
           [0.5],
           [0.6],
           [0.7],
           [0.8],
           [0.9],
           [1. ]])




```python
# scaler.fit(test_array)
test_array = scaler.transform(test_array)
```


```python
test_array
```




    array([[0.  ],
           [0.04],
           [0.08],
           [0.12],
           [0.16],
           [0.2 ]])




```python

```
