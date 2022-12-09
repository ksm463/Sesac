```python
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```


```python
cancer = load_breast_cancer(as_frame=True)
```


```python
cancer.data.head(2)
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
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.8</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.6</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.9</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.8</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 30 columns</p>
</div>




```python
cancer.target.head(2)
```




    0    0
    1    0
    Name: target, dtype: int32




```python
cancer.target_names
```




    array(['malignant', 'benign'], dtype='<U9')




```python
lr_clf = LogisticRegression(solver='liblinear')
knn_clf = KNeighborsClassifier(n_neighbors=8)
vo_clf = VotingClassifier([('lr',lr_clf),('knn',knn_clf)],voting='soft')
X_train,X_test,y_train,y_test = train_test_split(cancer.data,
                                                 cancer.target,
                                                 test_size=0.2,
                                                 random_state=156)
vo_clf.fit(X_train,y_train)
pred = vo_clf.predict(X_test)
accuracy_score(y_test,pred)
```




    0.956140350877193




```python
models = [lr_clf,knn_clf]
for model in models:
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    model_name = model.__class__.__name__
    print(f'{model_name} 정확도 : {accuracy_score(y_test,pred)}')
```

    LogisticRegression 정확도 : 0.9473684210526315
    KNeighborsClassifier 정확도 : 0.9385964912280702
    

    C:\Users\user\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    


```python
# 랜덤포레스트
```


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
def get_new_df(old_df):
    dup_df = pd.DataFrame(data=old_df.groupby('column_name').cumcount(),columns=['dup_cnt'])
    dup_df = dup_df.reset_index()
    new_df = pd.merge(old_df.reset_index(),dup_df,how='outer')
    new_df['column_name'] = new_df[['column_name','dup_cnt']].apply(lambda x: x[0]+'_'+str(x[1]) if x[1]>0 else x[0],axis=1)
    new_df.drop(columns=['index'],inplace=True)
    
    return new_df

def get_human_dataset():
    feature_name_df = pd.read_csv('human_activity/features.txt',
                                sep='\s+',
                                header=None,
                                names=['column_index','column_name'])
    name_df = get_new_df(feature_name_df)
    feature_name = name_df.iloc[:,1].values.tolist()
    X_train = pd.read_csv('human_activity/train/X_train.txt',sep='\s+',names=feature_name)
    X_test = pd.read_csv('human_activity/test/X_test.txt',sep='\s+',names=feature_name)
    y_train = pd.read_csv('human_activity/train/y_train.txt',sep='\s+',names=['action'])
    y_test = pd.read_csv('human_activity/test/y_test.txt',sep='\s+',names=['action'])
    return X_train,X_test,y_train,y_test
```


```python
X_train,X_test,y_train,y_test = get_human_dataset()
```


```python
rf_clf = RandomForestClassifier(random_state=0,max_depth=8)
rf_clf.fit(X_train,y_train)
pred = rf_clf.predict(X_test)
accuracy_score(y_test,pred)
```

    C:\Users\user\AppData\Local\Temp\ipykernel_18236\990638652.py:2: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf_clf.fit(X_train,y_train)
    




    0.9195792331184255




```python
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train,y_train)
pred = rf_clf.predict(X_test)
accuracy_score(y_test,pred)
```

    C:\Users\user\AppData\Local\Temp\ipykernel_18236\1442801055.py:2: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf_clf.fit(X_train,y_train)
    




    0.9253478113335596




```python
from sklearn.model_selection import GridSearchCV
```


```python
params ={
    'max_depth':[8,16,24],
    'min_samples_split':[2,8,16],
    'min_samples_leaf':[1,6,12]
}
```


```python
%%time
rf_clf = RandomForestClassifier(random_state=0,n_jobs=-1)
grid_cv = GridSearchCV(rf_clf,params,cv=2,n_jobs=-1)
grid_cv.fit(X_train,y_train)
```

    C:\Users\user\anaconda3\lib\site-packages\sklearn\model_selection\_search.py:926: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      self.best_estimator_.fit(X, y, **fit_params)
    

    Wall time: 36.6 s
    




    GridSearchCV(cv=2, estimator=RandomForestClassifier(n_jobs=-1, random_state=0),
                 n_jobs=-1,
                 param_grid={'max_depth': [8, 16, 24],
                             'min_samples_leaf': [1, 6, 12],
                             'min_samples_split': [2, 8, 16]})




```python
grid_cv.best_params_
```




    {'max_depth': 16, 'min_samples_leaf': 6, 'min_samples_split': 2}




```python
grid_cv.best_score_
```




    0.9164853101196953




```python
rf_clf = RandomForestClassifier(random_state=0,max_depth=16,min_samples_leaf=6,min_samples_split=2)
rf_clf.fit(X_train,y_train)
pred = rf_clf.predict(X_test)
accuracy_score(y_test,pred)
```

    C:\Users\user\AppData\Local\Temp\ipykernel_18236\795056095.py:2: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf_clf.fit(X_train,y_train)
    




    0.9260264675941635




```python

```
