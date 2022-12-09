```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
```


```python
dt_clf = DecisionTreeClassifier(random_state=156)
iris = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=11)
dt_clf.fit(X_train,y_train)
```




    DecisionTreeClassifier(random_state=156)




```python
from sklearn.tree import export_graphviz
```


```python
export_graphviz(dt_clf,
                 'iris.dot',
                 class_names=iris.target_names,
                 feature_names=iris.feature_names,
                 filled=True)
```


```python
import graphviz
```


```python
with open('iris.dot') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)
```




    
![svg](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_5_0.svg)
    




```python
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)

export_graphviz(dt_clf,
                 'iris1.dot',
                 class_names=iris.target_names,
                 feature_names=iris.feature_names,
                 filled=True)

with open('iris1.dot') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)
```




    
![svg](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_6_0.svg)
    




```python
dt_clf = DecisionTreeClassifier(random_state=156,max_depth=2)
dt_clf.fit(X_train,y_train)

export_graphviz(dt_clf,
                 'iris1.dot',
                 class_names=iris.target_names,
                 feature_names=iris.feature_names,
                 filled=True)

with open('iris1.dot') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)
```




    
![svg](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_7_0.svg)
    




```python
dt_clf = DecisionTreeClassifier(random_state=156,max_depth=3)
dt_clf.fit(X_train,y_train)

export_graphviz(dt_clf,
                 'iris1.dot',
                 class_names=iris.target_names,
                 feature_names=iris.feature_names,
                 filled=True)

with open('iris1.dot') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)
```




    
![svg](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_8_0.svg)
    




```python
dt_clf = DecisionTreeClassifier(random_state=156,min_samples_split=4)
dt_clf.fit(X_train,y_train)

export_graphviz(dt_clf,
                 'iris1.dot',
                 class_names=iris.target_names,
                 feature_names=iris.feature_names,
                 filled=True)

with open('iris1.dot') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)
```




    
![svg](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_9_0.svg)
    




```python
dt_clf = DecisionTreeClassifier(random_state=156,min_samples_split=5)
dt_clf.fit(X_train,y_train)

export_graphviz(dt_clf,
                 'iris1.dot',
                 class_names=iris.target_names,
                 feature_names=iris.feature_names,
                 filled=True)

with open('iris1.dot') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)
```




    
![svg](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_10_0.svg)
    




```python
dt_clf = DecisionTreeClassifier(random_state=156,min_samples_leaf=1)
dt_clf.fit(X_train,y_train)

export_graphviz(dt_clf,
                 'iris1.dot',
                 class_names=iris.target_names,
                 feature_names=iris.feature_names,
                 filled=True)

with open('iris1.dot') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)
```




    
![svg](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_11_0.svg)
    




```python
dt_clf = DecisionTreeClassifier(random_state=156,min_samples_leaf=2)
dt_clf.fit(X_train,y_train)

export_graphviz(dt_clf,
                 'iris1.dot',
                 class_names=iris.target_names,
                 feature_names=iris.feature_names,
                 filled=True)

with open('iris1.dot') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)
```




    
![svg](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_12_0.svg)
    




```python
dt_clf = DecisionTreeClassifier(random_state=156,min_samples_leaf=3)
dt_clf.fit(X_train,y_train)

export_graphviz(dt_clf,
                 'iris1.dot',
                 class_names=iris.target_names,
                 feature_names=iris.feature_names,
                 filled=True)

with open('iris1.dot') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)
```




    
![svg](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_13_0.svg)
    




```python
dt_clf = DecisionTreeClassifier(random_state=156,max_features=None)
dt_clf.fit(X_train,y_train)

export_graphviz(dt_clf,
                 'iris1.dot',
                 class_names=iris.target_names,
                 feature_names=iris.feature_names,
                 filled=True)

with open('iris1.dot') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)
```




    
![svg](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_14_0.svg)
    




```python
dt_clf = DecisionTreeClassifier(random_state=156,max_features=2)
dt_clf.fit(X_train,y_train)

export_graphviz(dt_clf,
                 'iris1.dot',
                 class_names=iris.target_names,
                 feature_names=iris.feature_names,
                 filled=True)

with open('iris1.dot') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)
```




    
![svg](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_15_0.svg)
    




```python
dt_clf.feature_importances_
```




    array([0.06005112, 0.        , 0.84613678, 0.0938121 ])




```python
iris.feature_names
```




    ['sepal length (cm)',
     'sepal width (cm)',
     'petal length (cm)',
     'petal width (cm)']




```python
import seaborn as sns
```


```python
sns.barplot(x=dt_clf.feature_importances_, y=iris.feature_names)
```




    <AxesSubplot:>




    
![png](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_19_1.png)
    



```python
import func01
import pandas as pd
```


```python
df = pd.read_csv('titanic.csv')
y = df['Survived']
X = df.drop(columns=['Survived'])
X = func01.transform_features(X)
```

    ['female' 'male']
    ['A' 'B' 'C' 'D' 'E' 'F' 'G' 'N' 'T']
    ['C' 'N' 'Q' 'S']
    


```python
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=11)
```


```python
dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train,y_train)
export_graphviz(dt_clf,
                 'titanic.dot',
                 class_names=['사망','생존'],
                 feature_names=X_train.columns,
                 filled=True)

with open('titanic.dot',encoding='utf-8') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)
```




    
![svg](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_23_0.svg)
    




```python
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
```


```python
X, y = make_classification(n_features=2,n_redundant=0,n_classes=3,n_clusters_per_class=1,random_state=0)
```


```python
y
```




    array([0, 1, 1, 1, 2, 2, 1, 0, 2, 2, 0, 1, 2, 1, 2, 0, 0, 1, 2, 1, 2, 0,
           1, 2, 1, 0, 2, 2, 2, 0, 2, 1, 1, 0, 1, 0, 0, 2, 1, 0, 0, 1, 0, 2,
           0, 2, 1, 0, 2, 0, 2, 2, 2, 1, 0, 1, 1, 0, 2, 0, 2, 0, 0, 2, 1, 1,
           0, 1, 1, 2, 1, 0, 2, 2, 2, 0, 0, 1, 1, 0, 2, 1, 2, 1, 0, 2, 1, 1,
           1, 1, 0, 0, 1, 0, 2, 2, 0, 2, 0, 0])




```python
plt.scatter(X[:,0],X[:,1])
```




    <matplotlib.collections.PathCollection at 0x229f951d280>




    
![png](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_27_1.png)
    



```python
plt.scatter(X[:,0],X[:,1],c=y)
```




    <matplotlib.collections.PathCollection at 0x229f95af4c0>




    
![png](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_28_1.png)
    



```python
plt.rcParams['axes.unicode_minus']=False
plt.scatter(X[:,0],X[:,1],c=y,edgecolors='k',s=25)
```




    <matplotlib.collections.PathCollection at 0x229f9789940>




    
![png](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_29_1.png)
    



```python
dt_clf = DecisionTreeClassifier(random_state=156).fit(X,y)
```


```python
func01.visualize_boundary(dt_clf,X,y)
```

    C:\pythonDir\05_machine_learning\func01.py:53: UserWarning: The following kwargs were not used by contour: 'clim'
      contours = ax.contourf(xx, yy, Z, alpha=0.3,
    


    
![png](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_31_1.png)
    



```python
dt_clf = DecisionTreeClassifier(min_samples_leaf=6,random_state=156).fit(X,y)
func01.visualize_boundary(dt_clf,X,y)
```

    C:\pythonDir\05_machine_learning\func01.py:53: UserWarning: The following kwargs were not used by contour: 'clim'
      contours = ax.contourf(xx, yy, Z, alpha=0.3,
    


    
![png](04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_files/04_01_%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%EC%8B%9C%EA%B0%81%ED%99%94_32_1.png)
    

