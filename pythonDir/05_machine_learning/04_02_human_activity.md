```python
import pandas as pd
import matplotlib.pyplot as plt
```


```python
pd.read_csv('human_activity/features.txt',sep='\s+')
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
      <th>1</th>
      <th>tBodyAcc-mean()-X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>tBodyAcc-mean()-Y</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>tBodyAcc-mean()-Z</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>tBodyAcc-std()-X</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>tBodyAcc-std()-Y</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>tBodyAcc-std()-Z</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>555</th>
      <td>557</td>
      <td>angle(tBodyGyroMean,gravityMean)</td>
    </tr>
    <tr>
      <th>556</th>
      <td>558</td>
      <td>angle(tBodyGyroJerkMean,gravityMean)</td>
    </tr>
    <tr>
      <th>557</th>
      <td>559</td>
      <td>angle(X,gravityMean)</td>
    </tr>
    <tr>
      <th>558</th>
      <td>560</td>
      <td>angle(Y,gravityMean)</td>
    </tr>
    <tr>
      <th>559</th>
      <td>561</td>
      <td>angle(Z,gravityMean)</td>
    </tr>
  </tbody>
</table>
<p>560 rows × 2 columns</p>
</div>




```python
pd.read_csv('human_activity/features.txt',
            sep='\s+',
            header=None,
            names=['column_index','column_name'])
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
      <th>column_index</th>
      <th>column_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>tBodyAcc-mean()-X</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>tBodyAcc-mean()-Y</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>tBodyAcc-mean()-Z</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>tBodyAcc-std()-X</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>tBodyAcc-std()-Y</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>556</th>
      <td>557</td>
      <td>angle(tBodyGyroMean,gravityMean)</td>
    </tr>
    <tr>
      <th>557</th>
      <td>558</td>
      <td>angle(tBodyGyroJerkMean,gravityMean)</td>
    </tr>
    <tr>
      <th>558</th>
      <td>559</td>
      <td>angle(X,gravityMean)</td>
    </tr>
    <tr>
      <th>559</th>
      <td>560</td>
      <td>angle(Y,gravityMean)</td>
    </tr>
    <tr>
      <th>560</th>
      <td>561</td>
      <td>angle(Z,gravityMean)</td>
    </tr>
  </tbody>
</table>
<p>561 rows × 2 columns</p>
</div>




```python
feature_name_df = pd.read_csv('human_activity/features.txt',
                                sep='\s+',
                                header=None,
                                names=['column_index','column_name'])
```


```python
feature_name_df.head(2)
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
      <th>column_index</th>
      <th>column_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>tBodyAcc-mean()-X</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>tBodyAcc-mean()-Y</td>
    </tr>
  </tbody>
</table>
</div>




```python
feature_name_df.iloc[:,1].values.tolist()
```




    ['tBodyAcc-mean()-X',
     'tBodyAcc-mean()-Y',
     'tBodyAcc-mean()-Z',
     'tBodyAcc-std()-X',
     'tBodyAcc-std()-Y',
     'tBodyAcc-std()-Z',
     'tBodyAcc-mad()-X',
     'tBodyAcc-mad()-Y',
     'tBodyAcc-mad()-Z',
     'tBodyAcc-max()-X',
     'tBodyAcc-max()-Y',
     'tBodyAcc-max()-Z',
     'tBodyAcc-min()-X',
     'tBodyAcc-min()-Y',
     'tBodyAcc-min()-Z',
     'tBodyAcc-sma()',
     'tBodyAcc-energy()-X',
     'tBodyAcc-energy()-Y',
     'tBodyAcc-energy()-Z',
     'tBodyAcc-iqr()-X',
     'tBodyAcc-iqr()-Y',
     'tBodyAcc-iqr()-Z',
     'tBodyAcc-entropy()-X',
     'tBodyAcc-entropy()-Y',
     'tBodyAcc-entropy()-Z',
     'tBodyAcc-arCoeff()-X,1',
     'tBodyAcc-arCoeff()-X,2',
     'tBodyAcc-arCoeff()-X,3',
     'tBodyAcc-arCoeff()-X,4',
     'tBodyAcc-arCoeff()-Y,1',
     'tBodyAcc-arCoeff()-Y,2',
     'tBodyAcc-arCoeff()-Y,3',
     'tBodyAcc-arCoeff()-Y,4',
     'tBodyAcc-arCoeff()-Z,1',
     'tBodyAcc-arCoeff()-Z,2',
     'tBodyAcc-arCoeff()-Z,3',
     'tBodyAcc-arCoeff()-Z,4',
     'tBodyAcc-correlation()-X,Y',
     'tBodyAcc-correlation()-X,Z',
     'tBodyAcc-correlation()-Y,Z',
     'tGravityAcc-mean()-X',
     'tGravityAcc-mean()-Y',
     'tGravityAcc-mean()-Z',
     'tGravityAcc-std()-X',
     'tGravityAcc-std()-Y',
     'tGravityAcc-std()-Z',
     'tGravityAcc-mad()-X',
     'tGravityAcc-mad()-Y',
     'tGravityAcc-mad()-Z',
     'tGravityAcc-max()-X',
     'tGravityAcc-max()-Y',
     'tGravityAcc-max()-Z',
     'tGravityAcc-min()-X',
     'tGravityAcc-min()-Y',
     'tGravityAcc-min()-Z',
     'tGravityAcc-sma()',
     'tGravityAcc-energy()-X',
     'tGravityAcc-energy()-Y',
     'tGravityAcc-energy()-Z',
     'tGravityAcc-iqr()-X',
     'tGravityAcc-iqr()-Y',
     'tGravityAcc-iqr()-Z',
     'tGravityAcc-entropy()-X',
     'tGravityAcc-entropy()-Y',
     'tGravityAcc-entropy()-Z',
     'tGravityAcc-arCoeff()-X,1',
     'tGravityAcc-arCoeff()-X,2',
     'tGravityAcc-arCoeff()-X,3',
     'tGravityAcc-arCoeff()-X,4',
     'tGravityAcc-arCoeff()-Y,1',
     'tGravityAcc-arCoeff()-Y,2',
     'tGravityAcc-arCoeff()-Y,3',
     'tGravityAcc-arCoeff()-Y,4',
     'tGravityAcc-arCoeff()-Z,1',
     'tGravityAcc-arCoeff()-Z,2',
     'tGravityAcc-arCoeff()-Z,3',
     'tGravityAcc-arCoeff()-Z,4',
     'tGravityAcc-correlation()-X,Y',
     'tGravityAcc-correlation()-X,Z',
     'tGravityAcc-correlation()-Y,Z',
     'tBodyAccJerk-mean()-X',
     'tBodyAccJerk-mean()-Y',
     'tBodyAccJerk-mean()-Z',
     'tBodyAccJerk-std()-X',
     'tBodyAccJerk-std()-Y',
     'tBodyAccJerk-std()-Z',
     'tBodyAccJerk-mad()-X',
     'tBodyAccJerk-mad()-Y',
     'tBodyAccJerk-mad()-Z',
     'tBodyAccJerk-max()-X',
     'tBodyAccJerk-max()-Y',
     'tBodyAccJerk-max()-Z',
     'tBodyAccJerk-min()-X',
     'tBodyAccJerk-min()-Y',
     'tBodyAccJerk-min()-Z',
     'tBodyAccJerk-sma()',
     'tBodyAccJerk-energy()-X',
     'tBodyAccJerk-energy()-Y',
     'tBodyAccJerk-energy()-Z',
     'tBodyAccJerk-iqr()-X',
     'tBodyAccJerk-iqr()-Y',
     'tBodyAccJerk-iqr()-Z',
     'tBodyAccJerk-entropy()-X',
     'tBodyAccJerk-entropy()-Y',
     'tBodyAccJerk-entropy()-Z',
     'tBodyAccJerk-arCoeff()-X,1',
     'tBodyAccJerk-arCoeff()-X,2',
     'tBodyAccJerk-arCoeff()-X,3',
     'tBodyAccJerk-arCoeff()-X,4',
     'tBodyAccJerk-arCoeff()-Y,1',
     'tBodyAccJerk-arCoeff()-Y,2',
     'tBodyAccJerk-arCoeff()-Y,3',
     'tBodyAccJerk-arCoeff()-Y,4',
     'tBodyAccJerk-arCoeff()-Z,1',
     'tBodyAccJerk-arCoeff()-Z,2',
     'tBodyAccJerk-arCoeff()-Z,3',
     'tBodyAccJerk-arCoeff()-Z,4',
     'tBodyAccJerk-correlation()-X,Y',
     'tBodyAccJerk-correlation()-X,Z',
     'tBodyAccJerk-correlation()-Y,Z',
     'tBodyGyro-mean()-X',
     'tBodyGyro-mean()-Y',
     'tBodyGyro-mean()-Z',
     'tBodyGyro-std()-X',
     'tBodyGyro-std()-Y',
     'tBodyGyro-std()-Z',
     'tBodyGyro-mad()-X',
     'tBodyGyro-mad()-Y',
     'tBodyGyro-mad()-Z',
     'tBodyGyro-max()-X',
     'tBodyGyro-max()-Y',
     'tBodyGyro-max()-Z',
     'tBodyGyro-min()-X',
     'tBodyGyro-min()-Y',
     'tBodyGyro-min()-Z',
     'tBodyGyro-sma()',
     'tBodyGyro-energy()-X',
     'tBodyGyro-energy()-Y',
     'tBodyGyro-energy()-Z',
     'tBodyGyro-iqr()-X',
     'tBodyGyro-iqr()-Y',
     'tBodyGyro-iqr()-Z',
     'tBodyGyro-entropy()-X',
     'tBodyGyro-entropy()-Y',
     'tBodyGyro-entropy()-Z',
     'tBodyGyro-arCoeff()-X,1',
     'tBodyGyro-arCoeff()-X,2',
     'tBodyGyro-arCoeff()-X,3',
     'tBodyGyro-arCoeff()-X,4',
     'tBodyGyro-arCoeff()-Y,1',
     'tBodyGyro-arCoeff()-Y,2',
     'tBodyGyro-arCoeff()-Y,3',
     'tBodyGyro-arCoeff()-Y,4',
     'tBodyGyro-arCoeff()-Z,1',
     'tBodyGyro-arCoeff()-Z,2',
     'tBodyGyro-arCoeff()-Z,3',
     'tBodyGyro-arCoeff()-Z,4',
     'tBodyGyro-correlation()-X,Y',
     'tBodyGyro-correlation()-X,Z',
     'tBodyGyro-correlation()-Y,Z',
     'tBodyGyroJerk-mean()-X',
     'tBodyGyroJerk-mean()-Y',
     'tBodyGyroJerk-mean()-Z',
     'tBodyGyroJerk-std()-X',
     'tBodyGyroJerk-std()-Y',
     'tBodyGyroJerk-std()-Z',
     'tBodyGyroJerk-mad()-X',
     'tBodyGyroJerk-mad()-Y',
     'tBodyGyroJerk-mad()-Z',
     'tBodyGyroJerk-max()-X',
     'tBodyGyroJerk-max()-Y',
     'tBodyGyroJerk-max()-Z',
     'tBodyGyroJerk-min()-X',
     'tBodyGyroJerk-min()-Y',
     'tBodyGyroJerk-min()-Z',
     'tBodyGyroJerk-sma()',
     'tBodyGyroJerk-energy()-X',
     'tBodyGyroJerk-energy()-Y',
     'tBodyGyroJerk-energy()-Z',
     'tBodyGyroJerk-iqr()-X',
     'tBodyGyroJerk-iqr()-Y',
     'tBodyGyroJerk-iqr()-Z',
     'tBodyGyroJerk-entropy()-X',
     'tBodyGyroJerk-entropy()-Y',
     'tBodyGyroJerk-entropy()-Z',
     'tBodyGyroJerk-arCoeff()-X,1',
     'tBodyGyroJerk-arCoeff()-X,2',
     'tBodyGyroJerk-arCoeff()-X,3',
     'tBodyGyroJerk-arCoeff()-X,4',
     'tBodyGyroJerk-arCoeff()-Y,1',
     'tBodyGyroJerk-arCoeff()-Y,2',
     'tBodyGyroJerk-arCoeff()-Y,3',
     'tBodyGyroJerk-arCoeff()-Y,4',
     'tBodyGyroJerk-arCoeff()-Z,1',
     'tBodyGyroJerk-arCoeff()-Z,2',
     'tBodyGyroJerk-arCoeff()-Z,3',
     'tBodyGyroJerk-arCoeff()-Z,4',
     'tBodyGyroJerk-correlation()-X,Y',
     'tBodyGyroJerk-correlation()-X,Z',
     'tBodyGyroJerk-correlation()-Y,Z',
     'tBodyAccMag-mean()',
     'tBodyAccMag-std()',
     'tBodyAccMag-mad()',
     'tBodyAccMag-max()',
     'tBodyAccMag-min()',
     'tBodyAccMag-sma()',
     'tBodyAccMag-energy()',
     'tBodyAccMag-iqr()',
     'tBodyAccMag-entropy()',
     'tBodyAccMag-arCoeff()1',
     'tBodyAccMag-arCoeff()2',
     'tBodyAccMag-arCoeff()3',
     'tBodyAccMag-arCoeff()4',
     'tGravityAccMag-mean()',
     'tGravityAccMag-std()',
     'tGravityAccMag-mad()',
     'tGravityAccMag-max()',
     'tGravityAccMag-min()',
     'tGravityAccMag-sma()',
     'tGravityAccMag-energy()',
     'tGravityAccMag-iqr()',
     'tGravityAccMag-entropy()',
     'tGravityAccMag-arCoeff()1',
     'tGravityAccMag-arCoeff()2',
     'tGravityAccMag-arCoeff()3',
     'tGravityAccMag-arCoeff()4',
     'tBodyAccJerkMag-mean()',
     'tBodyAccJerkMag-std()',
     'tBodyAccJerkMag-mad()',
     'tBodyAccJerkMag-max()',
     'tBodyAccJerkMag-min()',
     'tBodyAccJerkMag-sma()',
     'tBodyAccJerkMag-energy()',
     'tBodyAccJerkMag-iqr()',
     'tBodyAccJerkMag-entropy()',
     'tBodyAccJerkMag-arCoeff()1',
     'tBodyAccJerkMag-arCoeff()2',
     'tBodyAccJerkMag-arCoeff()3',
     'tBodyAccJerkMag-arCoeff()4',
     'tBodyGyroMag-mean()',
     'tBodyGyroMag-std()',
     'tBodyGyroMag-mad()',
     'tBodyGyroMag-max()',
     'tBodyGyroMag-min()',
     'tBodyGyroMag-sma()',
     'tBodyGyroMag-energy()',
     'tBodyGyroMag-iqr()',
     'tBodyGyroMag-entropy()',
     'tBodyGyroMag-arCoeff()1',
     'tBodyGyroMag-arCoeff()2',
     'tBodyGyroMag-arCoeff()3',
     'tBodyGyroMag-arCoeff()4',
     'tBodyGyroJerkMag-mean()',
     'tBodyGyroJerkMag-std()',
     'tBodyGyroJerkMag-mad()',
     'tBodyGyroJerkMag-max()',
     'tBodyGyroJerkMag-min()',
     'tBodyGyroJerkMag-sma()',
     'tBodyGyroJerkMag-energy()',
     'tBodyGyroJerkMag-iqr()',
     'tBodyGyroJerkMag-entropy()',
     'tBodyGyroJerkMag-arCoeff()1',
     'tBodyGyroJerkMag-arCoeff()2',
     'tBodyGyroJerkMag-arCoeff()3',
     'tBodyGyroJerkMag-arCoeff()4',
     'fBodyAcc-mean()-X',
     'fBodyAcc-mean()-Y',
     'fBodyAcc-mean()-Z',
     'fBodyAcc-std()-X',
     'fBodyAcc-std()-Y',
     'fBodyAcc-std()-Z',
     'fBodyAcc-mad()-X',
     'fBodyAcc-mad()-Y',
     'fBodyAcc-mad()-Z',
     'fBodyAcc-max()-X',
     'fBodyAcc-max()-Y',
     'fBodyAcc-max()-Z',
     'fBodyAcc-min()-X',
     'fBodyAcc-min()-Y',
     'fBodyAcc-min()-Z',
     'fBodyAcc-sma()',
     'fBodyAcc-energy()-X',
     'fBodyAcc-energy()-Y',
     'fBodyAcc-energy()-Z',
     'fBodyAcc-iqr()-X',
     'fBodyAcc-iqr()-Y',
     'fBodyAcc-iqr()-Z',
     'fBodyAcc-entropy()-X',
     'fBodyAcc-entropy()-Y',
     'fBodyAcc-entropy()-Z',
     'fBodyAcc-maxInds-X',
     'fBodyAcc-maxInds-Y',
     'fBodyAcc-maxInds-Z',
     'fBodyAcc-meanFreq()-X',
     'fBodyAcc-meanFreq()-Y',
     'fBodyAcc-meanFreq()-Z',
     'fBodyAcc-skewness()-X',
     'fBodyAcc-kurtosis()-X',
     'fBodyAcc-skewness()-Y',
     'fBodyAcc-kurtosis()-Y',
     'fBodyAcc-skewness()-Z',
     'fBodyAcc-kurtosis()-Z',
     'fBodyAcc-bandsEnergy()-1,8',
     'fBodyAcc-bandsEnergy()-9,16',
     'fBodyAcc-bandsEnergy()-17,24',
     'fBodyAcc-bandsEnergy()-25,32',
     'fBodyAcc-bandsEnergy()-33,40',
     'fBodyAcc-bandsEnergy()-41,48',
     'fBodyAcc-bandsEnergy()-49,56',
     'fBodyAcc-bandsEnergy()-57,64',
     'fBodyAcc-bandsEnergy()-1,16',
     'fBodyAcc-bandsEnergy()-17,32',
     'fBodyAcc-bandsEnergy()-33,48',
     'fBodyAcc-bandsEnergy()-49,64',
     'fBodyAcc-bandsEnergy()-1,24',
     'fBodyAcc-bandsEnergy()-25,48',
     'fBodyAcc-bandsEnergy()-1,8',
     'fBodyAcc-bandsEnergy()-9,16',
     'fBodyAcc-bandsEnergy()-17,24',
     'fBodyAcc-bandsEnergy()-25,32',
     'fBodyAcc-bandsEnergy()-33,40',
     'fBodyAcc-bandsEnergy()-41,48',
     'fBodyAcc-bandsEnergy()-49,56',
     'fBodyAcc-bandsEnergy()-57,64',
     'fBodyAcc-bandsEnergy()-1,16',
     'fBodyAcc-bandsEnergy()-17,32',
     'fBodyAcc-bandsEnergy()-33,48',
     'fBodyAcc-bandsEnergy()-49,64',
     'fBodyAcc-bandsEnergy()-1,24',
     'fBodyAcc-bandsEnergy()-25,48',
     'fBodyAcc-bandsEnergy()-1,8',
     'fBodyAcc-bandsEnergy()-9,16',
     'fBodyAcc-bandsEnergy()-17,24',
     'fBodyAcc-bandsEnergy()-25,32',
     'fBodyAcc-bandsEnergy()-33,40',
     'fBodyAcc-bandsEnergy()-41,48',
     'fBodyAcc-bandsEnergy()-49,56',
     'fBodyAcc-bandsEnergy()-57,64',
     'fBodyAcc-bandsEnergy()-1,16',
     'fBodyAcc-bandsEnergy()-17,32',
     'fBodyAcc-bandsEnergy()-33,48',
     'fBodyAcc-bandsEnergy()-49,64',
     'fBodyAcc-bandsEnergy()-1,24',
     'fBodyAcc-bandsEnergy()-25,48',
     'fBodyAccJerk-mean()-X',
     'fBodyAccJerk-mean()-Y',
     'fBodyAccJerk-mean()-Z',
     'fBodyAccJerk-std()-X',
     'fBodyAccJerk-std()-Y',
     'fBodyAccJerk-std()-Z',
     'fBodyAccJerk-mad()-X',
     'fBodyAccJerk-mad()-Y',
     'fBodyAccJerk-mad()-Z',
     'fBodyAccJerk-max()-X',
     'fBodyAccJerk-max()-Y',
     'fBodyAccJerk-max()-Z',
     'fBodyAccJerk-min()-X',
     'fBodyAccJerk-min()-Y',
     'fBodyAccJerk-min()-Z',
     'fBodyAccJerk-sma()',
     'fBodyAccJerk-energy()-X',
     'fBodyAccJerk-energy()-Y',
     'fBodyAccJerk-energy()-Z',
     'fBodyAccJerk-iqr()-X',
     'fBodyAccJerk-iqr()-Y',
     'fBodyAccJerk-iqr()-Z',
     'fBodyAccJerk-entropy()-X',
     'fBodyAccJerk-entropy()-Y',
     'fBodyAccJerk-entropy()-Z',
     'fBodyAccJerk-maxInds-X',
     'fBodyAccJerk-maxInds-Y',
     'fBodyAccJerk-maxInds-Z',
     'fBodyAccJerk-meanFreq()-X',
     'fBodyAccJerk-meanFreq()-Y',
     'fBodyAccJerk-meanFreq()-Z',
     'fBodyAccJerk-skewness()-X',
     'fBodyAccJerk-kurtosis()-X',
     'fBodyAccJerk-skewness()-Y',
     'fBodyAccJerk-kurtosis()-Y',
     'fBodyAccJerk-skewness()-Z',
     'fBodyAccJerk-kurtosis()-Z',
     'fBodyAccJerk-bandsEnergy()-1,8',
     'fBodyAccJerk-bandsEnergy()-9,16',
     'fBodyAccJerk-bandsEnergy()-17,24',
     'fBodyAccJerk-bandsEnergy()-25,32',
     'fBodyAccJerk-bandsEnergy()-33,40',
     'fBodyAccJerk-bandsEnergy()-41,48',
     'fBodyAccJerk-bandsEnergy()-49,56',
     'fBodyAccJerk-bandsEnergy()-57,64',
     'fBodyAccJerk-bandsEnergy()-1,16',
     'fBodyAccJerk-bandsEnergy()-17,32',
     'fBodyAccJerk-bandsEnergy()-33,48',
     'fBodyAccJerk-bandsEnergy()-49,64',
     'fBodyAccJerk-bandsEnergy()-1,24',
     'fBodyAccJerk-bandsEnergy()-25,48',
     'fBodyAccJerk-bandsEnergy()-1,8',
     'fBodyAccJerk-bandsEnergy()-9,16',
     'fBodyAccJerk-bandsEnergy()-17,24',
     'fBodyAccJerk-bandsEnergy()-25,32',
     'fBodyAccJerk-bandsEnergy()-33,40',
     'fBodyAccJerk-bandsEnergy()-41,48',
     'fBodyAccJerk-bandsEnergy()-49,56',
     'fBodyAccJerk-bandsEnergy()-57,64',
     'fBodyAccJerk-bandsEnergy()-1,16',
     'fBodyAccJerk-bandsEnergy()-17,32',
     'fBodyAccJerk-bandsEnergy()-33,48',
     'fBodyAccJerk-bandsEnergy()-49,64',
     'fBodyAccJerk-bandsEnergy()-1,24',
     'fBodyAccJerk-bandsEnergy()-25,48',
     'fBodyAccJerk-bandsEnergy()-1,8',
     'fBodyAccJerk-bandsEnergy()-9,16',
     'fBodyAccJerk-bandsEnergy()-17,24',
     'fBodyAccJerk-bandsEnergy()-25,32',
     'fBodyAccJerk-bandsEnergy()-33,40',
     'fBodyAccJerk-bandsEnergy()-41,48',
     'fBodyAccJerk-bandsEnergy()-49,56',
     'fBodyAccJerk-bandsEnergy()-57,64',
     'fBodyAccJerk-bandsEnergy()-1,16',
     'fBodyAccJerk-bandsEnergy()-17,32',
     'fBodyAccJerk-bandsEnergy()-33,48',
     'fBodyAccJerk-bandsEnergy()-49,64',
     'fBodyAccJerk-bandsEnergy()-1,24',
     'fBodyAccJerk-bandsEnergy()-25,48',
     'fBodyGyro-mean()-X',
     'fBodyGyro-mean()-Y',
     'fBodyGyro-mean()-Z',
     'fBodyGyro-std()-X',
     'fBodyGyro-std()-Y',
     'fBodyGyro-std()-Z',
     'fBodyGyro-mad()-X',
     'fBodyGyro-mad()-Y',
     'fBodyGyro-mad()-Z',
     'fBodyGyro-max()-X',
     'fBodyGyro-max()-Y',
     'fBodyGyro-max()-Z',
     'fBodyGyro-min()-X',
     'fBodyGyro-min()-Y',
     'fBodyGyro-min()-Z',
     'fBodyGyro-sma()',
     'fBodyGyro-energy()-X',
     'fBodyGyro-energy()-Y',
     'fBodyGyro-energy()-Z',
     'fBodyGyro-iqr()-X',
     'fBodyGyro-iqr()-Y',
     'fBodyGyro-iqr()-Z',
     'fBodyGyro-entropy()-X',
     'fBodyGyro-entropy()-Y',
     'fBodyGyro-entropy()-Z',
     'fBodyGyro-maxInds-X',
     'fBodyGyro-maxInds-Y',
     'fBodyGyro-maxInds-Z',
     'fBodyGyro-meanFreq()-X',
     'fBodyGyro-meanFreq()-Y',
     'fBodyGyro-meanFreq()-Z',
     'fBodyGyro-skewness()-X',
     'fBodyGyro-kurtosis()-X',
     'fBodyGyro-skewness()-Y',
     'fBodyGyro-kurtosis()-Y',
     'fBodyGyro-skewness()-Z',
     'fBodyGyro-kurtosis()-Z',
     'fBodyGyro-bandsEnergy()-1,8',
     'fBodyGyro-bandsEnergy()-9,16',
     'fBodyGyro-bandsEnergy()-17,24',
     'fBodyGyro-bandsEnergy()-25,32',
     'fBodyGyro-bandsEnergy()-33,40',
     'fBodyGyro-bandsEnergy()-41,48',
     'fBodyGyro-bandsEnergy()-49,56',
     'fBodyGyro-bandsEnergy()-57,64',
     'fBodyGyro-bandsEnergy()-1,16',
     'fBodyGyro-bandsEnergy()-17,32',
     'fBodyGyro-bandsEnergy()-33,48',
     'fBodyGyro-bandsEnergy()-49,64',
     'fBodyGyro-bandsEnergy()-1,24',
     'fBodyGyro-bandsEnergy()-25,48',
     'fBodyGyro-bandsEnergy()-1,8',
     'fBodyGyro-bandsEnergy()-9,16',
     'fBodyGyro-bandsEnergy()-17,24',
     'fBodyGyro-bandsEnergy()-25,32',
     'fBodyGyro-bandsEnergy()-33,40',
     'fBodyGyro-bandsEnergy()-41,48',
     'fBodyGyro-bandsEnergy()-49,56',
     'fBodyGyro-bandsEnergy()-57,64',
     'fBodyGyro-bandsEnergy()-1,16',
     'fBodyGyro-bandsEnergy()-17,32',
     'fBodyGyro-bandsEnergy()-33,48',
     'fBodyGyro-bandsEnergy()-49,64',
     'fBodyGyro-bandsEnergy()-1,24',
     'fBodyGyro-bandsEnergy()-25,48',
     'fBodyGyro-bandsEnergy()-1,8',
     'fBodyGyro-bandsEnergy()-9,16',
     'fBodyGyro-bandsEnergy()-17,24',
     'fBodyGyro-bandsEnergy()-25,32',
     'fBodyGyro-bandsEnergy()-33,40',
     'fBodyGyro-bandsEnergy()-41,48',
     'fBodyGyro-bandsEnergy()-49,56',
     'fBodyGyro-bandsEnergy()-57,64',
     'fBodyGyro-bandsEnergy()-1,16',
     'fBodyGyro-bandsEnergy()-17,32',
     'fBodyGyro-bandsEnergy()-33,48',
     'fBodyGyro-bandsEnergy()-49,64',
     'fBodyGyro-bandsEnergy()-1,24',
     'fBodyGyro-bandsEnergy()-25,48',
     'fBodyAccMag-mean()',
     'fBodyAccMag-std()',
     'fBodyAccMag-mad()',
     'fBodyAccMag-max()',
     'fBodyAccMag-min()',
     'fBodyAccMag-sma()',
     'fBodyAccMag-energy()',
     'fBodyAccMag-iqr()',
     'fBodyAccMag-entropy()',
     'fBodyAccMag-maxInds',
     'fBodyAccMag-meanFreq()',
     'fBodyAccMag-skewness()',
     'fBodyAccMag-kurtosis()',
     'fBodyBodyAccJerkMag-mean()',
     'fBodyBodyAccJerkMag-std()',
     'fBodyBodyAccJerkMag-mad()',
     'fBodyBodyAccJerkMag-max()',
     'fBodyBodyAccJerkMag-min()',
     'fBodyBodyAccJerkMag-sma()',
     'fBodyBodyAccJerkMag-energy()',
     'fBodyBodyAccJerkMag-iqr()',
     'fBodyBodyAccJerkMag-entropy()',
     'fBodyBodyAccJerkMag-maxInds',
     'fBodyBodyAccJerkMag-meanFreq()',
     'fBodyBodyAccJerkMag-skewness()',
     'fBodyBodyAccJerkMag-kurtosis()',
     'fBodyBodyGyroMag-mean()',
     'fBodyBodyGyroMag-std()',
     'fBodyBodyGyroMag-mad()',
     'fBodyBodyGyroMag-max()',
     'fBodyBodyGyroMag-min()',
     'fBodyBodyGyroMag-sma()',
     'fBodyBodyGyroMag-energy()',
     'fBodyBodyGyroMag-iqr()',
     'fBodyBodyGyroMag-entropy()',
     'fBodyBodyGyroMag-maxInds',
     'fBodyBodyGyroMag-meanFreq()',
     'fBodyBodyGyroMag-skewness()',
     'fBodyBodyGyroMag-kurtosis()',
     'fBodyBodyGyroJerkMag-mean()',
     'fBodyBodyGyroJerkMag-std()',
     'fBodyBodyGyroJerkMag-mad()',
     'fBodyBodyGyroJerkMag-max()',
     'fBodyBodyGyroJerkMag-min()',
     'fBodyBodyGyroJerkMag-sma()',
     'fBodyBodyGyroJerkMag-energy()',
     'fBodyBodyGyroJerkMag-iqr()',
     'fBodyBodyGyroJerkMag-entropy()',
     'fBodyBodyGyroJerkMag-maxInds',
     'fBodyBodyGyroJerkMag-meanFreq()',
     'fBodyBodyGyroJerkMag-skewness()',
     'fBodyBodyGyroJerkMag-kurtosis()',
     'angle(tBodyAccMean,gravity)',
     'angle(tBodyAccJerkMean),gravityMean)',
     'angle(tBodyGyroMean,gravityMean)',
     'angle(tBodyGyroJerkMean,gravityMean)',
     'angle(X,gravityMean)',
     'angle(Y,gravityMean)',
     'angle(Z,gravityMean)']




```python
feature_name = feature_name_df.iloc[:,1].values.tolist()
```


```python
feature_name[:10]
```




    ['tBodyAcc-mean()-X',
     'tBodyAcc-mean()-Y',
     'tBodyAcc-mean()-Z',
     'tBodyAcc-std()-X',
     'tBodyAcc-std()-Y',
     'tBodyAcc-std()-Z',
     'tBodyAcc-mad()-X',
     'tBodyAcc-mad()-Y',
     'tBodyAcc-mad()-Z',
     'tBodyAcc-max()-X']




```python
feature_name_df.groupby('column_name').count()
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
      <th>column_index</th>
    </tr>
    <tr>
      <th>column_name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>angle(X,gravityMean)</th>
      <td>1</td>
    </tr>
    <tr>
      <th>angle(Y,gravityMean)</th>
      <td>1</td>
    </tr>
    <tr>
      <th>angle(Z,gravityMean)</th>
      <td>1</td>
    </tr>
    <tr>
      <th>angle(tBodyAccJerkMean),gravityMean)</th>
      <td>1</td>
    </tr>
    <tr>
      <th>angle(tBodyAccMean,gravity)</th>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>tGravityAccMag-max()</th>
      <td>1</td>
    </tr>
    <tr>
      <th>tGravityAccMag-mean()</th>
      <td>1</td>
    </tr>
    <tr>
      <th>tGravityAccMag-min()</th>
      <td>1</td>
    </tr>
    <tr>
      <th>tGravityAccMag-sma()</th>
      <td>1</td>
    </tr>
    <tr>
      <th>tGravityAccMag-std()</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>477 rows × 1 columns</p>
</div>




```python
feature_dup_df = feature_name_df.groupby('column_name').count()
```


```python
feature_dup_df[feature_dup_df['column_index']>1].count()
```




    column_index    42
    dtype: int64




```python
def get_new_df(old_df):
    new_df = pd.DataFrame(data=old_df.groupby('column_name').cumcount(),columns=['dup_cnt'])
    return new_df
```


```python
get_new_df(feature_name_df)
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
      <th>dup_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>556</th>
      <td>0</td>
    </tr>
    <tr>
      <th>557</th>
      <td>0</td>
    </tr>
    <tr>
      <th>558</th>
      <td>0</td>
    </tr>
    <tr>
      <th>559</th>
      <td>0</td>
    </tr>
    <tr>
      <th>560</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>561 rows × 1 columns</p>
</div>




```python
df = get_new_df(feature_name_df)
```


```python
df[df['dup_cnt']>0]
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
      <th>dup_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>316</th>
      <td>1</td>
    </tr>
    <tr>
      <th>317</th>
      <td>1</td>
    </tr>
    <tr>
      <th>318</th>
      <td>1</td>
    </tr>
    <tr>
      <th>319</th>
      <td>1</td>
    </tr>
    <tr>
      <th>320</th>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>497</th>
      <td>2</td>
    </tr>
    <tr>
      <th>498</th>
      <td>2</td>
    </tr>
    <tr>
      <th>499</th>
      <td>2</td>
    </tr>
    <tr>
      <th>500</th>
      <td>2</td>
    </tr>
    <tr>
      <th>501</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>84 rows × 1 columns</p>
</div>




```python
def get_new_df(old_df):
    dup_df = pd.DataFrame(data=old_df.groupby('column_name').cumcount(),columns=['dup_cnt'])
    new_df = dup_df.reset_index()
    return new_df
```


```python
df = get_new_df(feature_name_df)
df
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
      <th>index</th>
      <th>dup_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>556</th>
      <td>556</td>
      <td>0</td>
    </tr>
    <tr>
      <th>557</th>
      <td>557</td>
      <td>0</td>
    </tr>
    <tr>
      <th>558</th>
      <td>558</td>
      <td>0</td>
    </tr>
    <tr>
      <th>559</th>
      <td>559</td>
      <td>0</td>
    </tr>
    <tr>
      <th>560</th>
      <td>560</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>561 rows × 2 columns</p>
</div>




```python
def get_new_df(old_df):
    dup_df = pd.DataFrame(data=old_df.groupby('column_name').cumcount(),columns=['dup_cnt'])
    dup_df = dup_df.reset_index()
    new_df = pd.merge(old_df.reset_index(),dup_df)
    
    return new_df
```


```python
df = get_new_df(feature_name_df)
df
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
      <th>index</th>
      <th>column_index</th>
      <th>column_name</th>
      <th>dup_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>tBodyAcc-mean()-X</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>tBodyAcc-mean()-Y</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>tBodyAcc-mean()-Z</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>tBodyAcc-std()-X</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>tBodyAcc-std()-Y</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>556</th>
      <td>556</td>
      <td>557</td>
      <td>angle(tBodyGyroMean,gravityMean)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>557</th>
      <td>557</td>
      <td>558</td>
      <td>angle(tBodyGyroJerkMean,gravityMean)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>558</th>
      <td>558</td>
      <td>559</td>
      <td>angle(X,gravityMean)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>559</th>
      <td>559</td>
      <td>560</td>
      <td>angle(Y,gravityMean)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>560</th>
      <td>560</td>
      <td>561</td>
      <td>angle(Z,gravityMean)</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>561 rows × 4 columns</p>
</div>




```python
def get_new_df(old_df):
    dup_df = pd.DataFrame(data=old_df.groupby('column_name').cumcount(),columns=['dup_cnt'])
    dup_df = dup_df.reset_index()
    new_df = pd.merge(old_df.reset_index(),dup_df,how='outer')
    new_df['column_name'] = new_df[['column_name','dup_cnt']].apply(lambda x: x[0]+'_'+str(x[1]) if x[1]>0 else x[0],axis=1)
    new_df.drop(columns=['index'],inplace=True)
    
    return new_df
```


```python
df = get_new_df(feature_name_df)
df
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
      <th>column_index</th>
      <th>column_name</th>
      <th>dup_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>tBodyAcc-mean()-X</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>tBodyAcc-mean()-Y</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>tBodyAcc-mean()-Z</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>tBodyAcc-std()-X</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>tBodyAcc-std()-Y</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>556</th>
      <td>557</td>
      <td>angle(tBodyGyroMean,gravityMean)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>557</th>
      <td>558</td>
      <td>angle(tBodyGyroJerkMean,gravityMean)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>558</th>
      <td>559</td>
      <td>angle(X,gravityMean)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>559</th>
      <td>560</td>
      <td>angle(Y,gravityMean)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>560</th>
      <td>561</td>
      <td>angle(Z,gravityMean)</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>561 rows × 3 columns</p>
</div>




```python
df[df['dup_cnt']>0]
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
      <th>column_index</th>
      <th>column_name</th>
      <th>dup_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>316</th>
      <td>317</td>
      <td>fBodyAcc-bandsEnergy()-1,8_1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>317</th>
      <td>318</td>
      <td>fBodyAcc-bandsEnergy()-9,16_1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>318</th>
      <td>319</td>
      <td>fBodyAcc-bandsEnergy()-17,24_1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>319</th>
      <td>320</td>
      <td>fBodyAcc-bandsEnergy()-25,32_1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>320</th>
      <td>321</td>
      <td>fBodyAcc-bandsEnergy()-33,40_1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>497</th>
      <td>498</td>
      <td>fBodyGyro-bandsEnergy()-17,32_2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>498</th>
      <td>499</td>
      <td>fBodyGyro-bandsEnergy()-33,48_2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>499</th>
      <td>500</td>
      <td>fBodyGyro-bandsEnergy()-49,64_2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>500</th>
      <td>501</td>
      <td>fBodyGyro-bandsEnergy()-1,24_2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>501</th>
      <td>502</td>
      <td>fBodyGyro-bandsEnergy()-25,48_2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>84 rows × 3 columns</p>
</div>




```python
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
X_train.head(2)
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
      <th>tBodyAcc-mean()-X</th>
      <th>tBodyAcc-mean()-Y</th>
      <th>tBodyAcc-mean()-Z</th>
      <th>tBodyAcc-std()-X</th>
      <th>tBodyAcc-std()-Y</th>
      <th>tBodyAcc-std()-Z</th>
      <th>tBodyAcc-mad()-X</th>
      <th>tBodyAcc-mad()-Y</th>
      <th>tBodyAcc-mad()-Z</th>
      <th>tBodyAcc-max()-X</th>
      <th>...</th>
      <th>fBodyBodyGyroJerkMag-meanFreq()</th>
      <th>fBodyBodyGyroJerkMag-skewness()</th>
      <th>fBodyBodyGyroJerkMag-kurtosis()</th>
      <th>angle(tBodyAccMean,gravity)</th>
      <th>angle(tBodyAccJerkMean),gravityMean)</th>
      <th>angle(tBodyGyroMean,gravityMean)</th>
      <th>angle(tBodyGyroJerkMean,gravityMean)</th>
      <th>angle(X,gravityMean)</th>
      <th>angle(Y,gravityMean)</th>
      <th>angle(Z,gravityMean)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.288585</td>
      <td>-0.020294</td>
      <td>-0.132905</td>
      <td>-0.995279</td>
      <td>-0.983111</td>
      <td>-0.913526</td>
      <td>-0.995112</td>
      <td>-0.983185</td>
      <td>-0.923527</td>
      <td>-0.934724</td>
      <td>...</td>
      <td>-0.074323</td>
      <td>-0.298676</td>
      <td>-0.710304</td>
      <td>-0.112754</td>
      <td>0.030400</td>
      <td>-0.464761</td>
      <td>-0.018446</td>
      <td>-0.841247</td>
      <td>0.179941</td>
      <td>-0.058627</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.278419</td>
      <td>-0.016411</td>
      <td>-0.123520</td>
      <td>-0.998245</td>
      <td>-0.975300</td>
      <td>-0.960322</td>
      <td>-0.998807</td>
      <td>-0.974914</td>
      <td>-0.957686</td>
      <td>-0.943068</td>
      <td>...</td>
      <td>0.158075</td>
      <td>-0.595051</td>
      <td>-0.861499</td>
      <td>0.053477</td>
      <td>-0.007435</td>
      <td>-0.732626</td>
      <td>0.703511</td>
      <td>-0.844788</td>
      <td>0.180289</td>
      <td>-0.054317</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 561 columns</p>
</div>




```python
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7352 entries, 0 to 7351
    Columns: 561 entries, tBodyAcc-mean()-X to angle(Z,gravityMean)
    dtypes: float64(561)
    memory usage: 31.5 MB
    


```python
y_train['action'].value_counts()
```




    6    1407
    5    1374
    4    1286
    1    1226
    2    1073
    3     986
    Name: action, dtype: int64




```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
```


```python
dt_clf = DecisionTreeClassifier(random_state=156)
dt_clf.fit(X_train,y_train)
pred = dt_clf.predict(X_test)
accuracy_score(y_test,pred)
```




    0.8547675602307431




```python
dt_clf.get_params()
```




    {'ccp_alpha': 0.0,
     'class_weight': None,
     'criterion': 'gini',
     'max_depth': None,
     'max_features': None,
     'max_leaf_nodes': None,
     'min_impurity_decrease': 0.0,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'random_state': 156,
     'splitter': 'best'}




```python
from sklearn.model_selection import GridSearchCV
```


```python
params = {
    'max_depth': [6,8,10,12,16,20,24],
    'min_samples_split': [16],
}

grid_cv = GridSearchCV(dt_clf,params,scoring='accuracy',cv=5,verbose=1)
```


```python
%%time
grid_cv.fit(X_train,y_train)
```

    Fitting 5 folds for each of 7 candidates, totalling 35 fits
    Wall time: 1min 6s
    




    GridSearchCV(cv=5, estimator=DecisionTreeClassifier(random_state=156),
                 param_grid={'max_depth': [6, 8, 10, 12, 16, 20, 24],
                             'min_samples_split': [16]},
                 scoring='accuracy', verbose=1)




```python
grid_cv.best_score_
```




    0.8548794147162603




```python
grid_cv.best_params_
```




    {'max_depth': 8, 'min_samples_split': 16}




```python
cv_result = pd.DataFrame(grid_cv.cv_results_)
```


```python
cv_result
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
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_max_depth</th>
      <th>param_min_samples_split</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.198689</td>
      <td>0.030454</td>
      <td>0.005280</td>
      <td>0.000692</td>
      <td>6</td>
      <td>16</td>
      <td>{'max_depth': 6, 'min_samples_split': 16}</td>
      <td>0.813732</td>
      <td>0.868117</td>
      <td>0.819728</td>
      <td>0.866667</td>
      <td>0.870068</td>
      <td>0.847662</td>
      <td>0.025350</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.495218</td>
      <td>0.041385</td>
      <td>0.005207</td>
      <td>0.000875</td>
      <td>8</td>
      <td>16</td>
      <td>{'max_depth': 8, 'min_samples_split': 16}</td>
      <td>0.806254</td>
      <td>0.830048</td>
      <td>0.860544</td>
      <td>0.874830</td>
      <td>0.902721</td>
      <td>0.854879</td>
      <td>0.033764</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.734049</td>
      <td>0.070149</td>
      <td>0.004999</td>
      <td>0.000122</td>
      <td>10</td>
      <td>16</td>
      <td>{'max_depth': 10, 'min_samples_split': 16}</td>
      <td>0.804895</td>
      <td>0.816451</td>
      <td>0.866667</td>
      <td>0.884354</td>
      <td>0.891156</td>
      <td>0.852705</td>
      <td>0.035427</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.936855</td>
      <td>0.098239</td>
      <td>0.004345</td>
      <td>0.000399</td>
      <td>12</td>
      <td>16</td>
      <td>{'max_depth': 12, 'min_samples_split': 16}</td>
      <td>0.798097</td>
      <td>0.810333</td>
      <td>0.851020</td>
      <td>0.884354</td>
      <td>0.885034</td>
      <td>0.845768</td>
      <td>0.036295</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.142756</td>
      <td>0.157106</td>
      <td>0.004955</td>
      <td>0.000388</td>
      <td>16</td>
      <td>16</td>
      <td>{'max_depth': 16, 'min_samples_split': 16}</td>
      <td>0.800816</td>
      <td>0.815092</td>
      <td>0.858503</td>
      <td>0.876871</td>
      <td>0.884354</td>
      <td>0.847127</td>
      <td>0.033379</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.210699</td>
      <td>0.196661</td>
      <td>0.005452</td>
      <td>0.000922</td>
      <td>20</td>
      <td>16</td>
      <td>{'max_depth': 20, 'min_samples_split': 16}</td>
      <td>0.798097</td>
      <td>0.815092</td>
      <td>0.858503</td>
      <td>0.876871</td>
      <td>0.894558</td>
      <td>0.848624</td>
      <td>0.036559</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.231004</td>
      <td>0.198891</td>
      <td>0.004594</td>
      <td>0.000562</td>
      <td>24</td>
      <td>16</td>
      <td>{'max_depth': 24, 'min_samples_split': 16}</td>
      <td>0.798097</td>
      <td>0.815092</td>
      <td>0.858503</td>
      <td>0.876871</td>
      <td>0.894558</td>
      <td>0.848624</td>
      <td>0.036559</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
cv_result[['param_max_depth','mean_test_score']]
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
      <th>param_max_depth</th>
      <th>mean_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>0.847662</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>0.854879</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>0.852705</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>0.845768</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>0.847127</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20</td>
      <td>0.848624</td>
    </tr>
    <tr>
      <th>6</th>
      <td>24</td>
      <td>0.848624</td>
    </tr>
  </tbody>
</table>
</div>




```python
max_depth = [6,8,10,12,16,20,24]
for depth in max_depth:
    dt_clf = DecisionTreeClassifier(max_depth=depth,min_samples_split=16,random_state=156)
    dt_clf.fit(X_train,y_train)
    pred = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test,pred)
    print(f'max_depth:{depth} 정확도:{accuracy:.4f}')
```

    max_depth:6 정확도:0.8551
    max_depth:8 정확도:0.8717
    max_depth:10 정확도:0.8599
    max_depth:12 정확도:0.8571
    max_depth:16 정확도:0.8599
    max_depth:20 정확도:0.8565
    max_depth:24 정확도:0.8565
    


```python
%%time
params = {
    'max_depth': [8,12,16,20],
    'min_samples_split': [16,24],
}

grid_cv = GridSearchCV(dt_clf,params,scoring='accuracy',cv=5,verbose=1)

grid_cv.fit(X_train,y_train)
```

    Fitting 5 folds for each of 8 candidates, totalling 40 fits
    Wall time: 1min 20s
    




    GridSearchCV(cv=5,
                 estimator=DecisionTreeClassifier(max_depth=24,
                                                  min_samples_split=16,
                                                  random_state=156),
                 param_grid={'max_depth': [8, 12, 16, 20],
                             'min_samples_split': [16, 24]},
                 scoring='accuracy', verbose=1)




```python
grid_cv.best_params_
```




    {'max_depth': 8, 'min_samples_split': 16}




```python
grid_cv.best_score_
```




    0.8548794147162603




```python
%%time
params = {
    'max_depth': [8],
    'min_samples_split': [8,12,16,20,24],
}

grid_cv = GridSearchCV(dt_clf,params,scoring='accuracy',cv=5,verbose=1)

grid_cv.fit(X_train,y_train)
```

    Fitting 5 folds for each of 5 candidates, totalling 25 fits
    


```python
grid_cv.best_params_
```


```python
grid_cv.best_score_
```


```python
pred = grid_cv.best_estimator_.predict(X_test)
accuracy_score(y_test,pred)
```


```python
grid_cv.best_estimator_.feature_importances_
```


```python
pd.Series(grid_cv.best_estimator_.feature_importances_)
```


```python
data = pd.Series(grid_cv.best_estimator_.feature_importances_,index=X_train.columns)
```


```python
data.sort_values(ascending=False)
```


```python
top10 = data.sort_values(ascending=False)[:10]
```


```python
import seaborn as sns
```


```python
sns.barplot(x=top10,y=top10.index)
```
