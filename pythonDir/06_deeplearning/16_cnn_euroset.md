## Euroset
* 실습용 데이터


```python
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
```


```python
data_dir = 'dataset/'

tfds.load('eurosat/rgb', split=['train[:80%]','train[80%:]'],
          shuffle_files=True, # 내용을 섞어서 가져욤
          as_supervised=True, # 딕셔너리 형식으로 가져옴
          with_info=True, # 정보도 같이 가져옴
          data_dir=data_dir) # 가져올 위치
```

    Downloading and preparing dataset 89.91 MiB (download: 89.91 MiB, generated: Unknown size, total: 89.91 MiB) to dataset/eurosat/rgb/2.0.0...
    


    Dl Completed...: 0 url [00:00, ? url/s]



    Dl Size...: 0 MiB [00:00, ? MiB/s]



    Extraction completed...: 0 file [00:00, ? file/s]



    Generating splits...:   0%|          | 0/1 [00:00<?, ? splits/s]



    Generating train examples...:   0%|          | 0/27000 [00:00<?, ? examples/s]



    Shuffling dataset/eurosat/rgb/2.0.0.incompleteNKC38O/eurosat-train.tfrecord*...:   0%|          | 0/27000 [00:…


    Dataset eurosat downloaded and prepared to dataset/eurosat/rgb/2.0.0. Subsequent calls will reuse this data.
    




    ([<PrefetchDataset element_spec=(TensorSpec(shape=(64, 64, 3), dtype=tf.uint8, name=None), TensorSpec(shape=(), dtype=tf.int64, name=None))>,
      <PrefetchDataset element_spec=(TensorSpec(shape=(64, 64, 3), dtype=tf.uint8, name=None), TensorSpec(shape=(), dtype=tf.int64, name=None))>],
     tfds.core.DatasetInfo(
         name='eurosat',
         full_name='eurosat/rgb/2.0.0',
         description="""
         EuroSAT dataset is based on Sentinel-2 satellite images covering 13 spectral
         bands and consisting of 10 classes with 27000 labeled and
         geo-referenced samples.
         
         Two datasets are offered:
         - rgb: Contains only the optical R, G, B frequency bands encoded as JPEG image.
         - all: Contains all 13 bands in the original value range (float32).
         
         URL: https://github.com/phelber/eurosat
         """,
         config_description="""
         Sentinel-2 RGB channels
         """,
         homepage='https://github.com/phelber/eurosat',
         data_path='dataset/eurosat/rgb/2.0.0',
         file_format=tfrecord,
         download_size=89.91 MiB,
         dataset_size=89.50 MiB,
         features=FeaturesDict({
             'filename': Text(shape=(), dtype=tf.string),
             'image': Image(shape=(64, 64, 3), dtype=tf.uint8),
             'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=10),
         }),
         supervised_keys=('image', 'label'),
         disable_shuffling=False,
         splits={
             'train': <SplitInfo num_examples=27000, num_shards=1>,
         },
         citation="""@misc{helber2017eurosat,
             title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
             author={Patrick Helber and Benjamin Bischke and Andreas Dengel and Damian Borth},
             year={2017},
             eprint={1709.00029},
             archivePrefix={arXiv},
             primaryClass={cs.CV}
         }""",
     ))




```python
data_dir = 'dataset/'

(train_ds,valid_ds),info = tfds.load('eurosat/rgb', split=['train[:80%]','train[80%:]'],
                                            shuffle_files=True, # 내용을 섞어서 가져욤
                                            as_supervised=True, # 딕셔너리 형식으로 가져옴
                                            with_info=True, # 정보도 같이 가져옴
                                            data_dir=data_dir) # 가져올 위치
```


```python
train_ds,valid_ds
```




    (<PrefetchDataset element_spec=(TensorSpec(shape=(64, 64, 3), dtype=tf.uint8, name=None), TensorSpec(shape=(), dtype=tf.int64, name=None))>,
     <PrefetchDataset element_spec=(TensorSpec(shape=(64, 64, 3), dtype=tf.uint8, name=None), TensorSpec(shape=(), dtype=tf.int64, name=None))>)




```python
info
```




    tfds.core.DatasetInfo(
        name='eurosat',
        full_name='eurosat/rgb/2.0.0',
        description="""
        EuroSAT dataset is based on Sentinel-2 satellite images covering 13 spectral
        bands and consisting of 10 classes with 27000 labeled and
        geo-referenced samples.
        
        Two datasets are offered:
        - rgb: Contains only the optical R, G, B frequency bands encoded as JPEG image.
        - all: Contains all 13 bands in the original value range (float32).
        
        URL: https://github.com/phelber/eurosat
        """,
        config_description="""
        Sentinel-2 RGB channels
        """,
        homepage='https://github.com/phelber/eurosat',
        data_path='dataset/eurosat/rgb/2.0.0',
        file_format=tfrecord,
        download_size=89.91 MiB,
        dataset_size=89.50 MiB,
        features=FeaturesDict({
            'filename': Text(shape=(), dtype=tf.string),
            'image': Image(shape=(64, 64, 3), dtype=tf.uint8),
            'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=10),
        }),
        supervised_keys=('image', 'label'),
        disable_shuffling=False,
        splits={
            'train': <SplitInfo num_examples=27000, num_shards=1>,
        },
        citation="""@misc{helber2017eurosat,
            title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
            author={Patrick Helber and Benjamin Bischke and Andreas Dengel and Damian Borth},
            year={2017},
            eprint={1709.00029},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
        }""",
    )




```python
tfds.show_examples(train_ds,info)
```


    
![png](16_cnn_euroset_files/16_cnn_euroset_6_0.png)
    





    
![png](16_cnn_euroset_files/16_cnn_euroset_6_1.png)
    




```python
tfds.as_dataframe(valid_ds.take(5),info)
```





  <div id="df-9905e4bb-cd05-4b87-9763-93f6151259ee">
    <div class="colab-df-container">
      <style type="text/css">
</style>
<table id="T_c883c_">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >image</th>
      <th class="col_heading level0 col1" >label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_c883c_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_c883c_row0_col0" class="data row0 col0" ><img src="data:image/png;base64,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" alt="Img" /></td>
      <td id="T_c883c_row0_col1" class="data row0 col1" >5 (Pasture)</td>
    </tr>
    <tr>
      <th id="T_c883c_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_c883c_row1_col0" class="data row1 col0" ><img src="data:image/png;base64,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" alt="Img" /></td>
      <td id="T_c883c_row1_col1" class="data row1 col1" >7 (Residential)</td>
    </tr>
    <tr>
      <th id="T_c883c_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_c883c_row2_col0" class="data row2 col0" ><img src="data:image/png;base64,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" alt="Img" /></td>
      <td id="T_c883c_row2_col1" class="data row2 col1" >0 (AnnualCrop)</td>
    </tr>
    <tr>
      <th id="T_c883c_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_c883c_row3_col0" class="data row3 col0" ><img src="data:image/png;base64,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" alt="Img" /></td>
      <td id="T_c883c_row3_col1" class="data row3 col1" >1 (Forest)</td>
    </tr>
    <tr>
      <th id="T_c883c_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_c883c_row4_col0" class="data row4 col0" ><img src="data:image/png;base64,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" alt="Img" /></td>
      <td id="T_c883c_row4_col1" class="data row4 col1" >0 (AnnualCrop)</td>
    </tr>
  </tbody>
</table>

      <button class="colab-df-convert" onclick="convertToInteractive('df-9905e4bb-cd05-4b87-9763-93f6151259ee')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9905e4bb-cd05-4b87-9763-93f6151259ee button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9905e4bb-cd05-4b87-9763-93f6151259ee');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
num_classes = info.features['label'].num_classes
num_classes
```




    10




```python
# 개별 데이터값 확인
print(info.features['label'].int2str(7))
```

    Residential
    


```python
batch_size=64 # 한번에 올리는 데이터 숫자
buffer_size=1000 # 미리 준비해두는 데이터 단위. 배치사이즈수량이 빠져나가면 다시 원데이터에서 같은 수량을 가져온다.

def preprocess_data(image,label):
  image = tf.cast(image,tf.float32)/255.
  return image, label

train_data = train_ds.map(preprocess_data,num_parallel_calls=tf.data.AUTOTUNE) # 오토튠은 데이터를 제공할 때 병렬처리를 해줌. 
valid_data = valid_ds.map(preprocess_data,num_parallel_calls=tf.data.AUTOTUNE)

train_data = train_data.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
valid_data = valid_data.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE) # 섞을 필요가 없어 셔플 미사용.
```


```python
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten,Conv2D,MaxPooling2D,BatchNormalization

def build_model():
  model = Sequential([
      BatchNormalization(),
      Conv2D(32,(3,3), padding='same',activation='relu'),
      MaxPooling2D((2,2)),

      BatchNormalization(),
      Conv2D(64,(3,3), padding='same',activation='relu'),
      MaxPooling2D((2,2)),

      Flatten(),
      Dense(128,activation='relu'),
      Dropout(0.3),

      Dense(64,activation='relu'),
      Dropout(0.3),
      Dense(num_classes,activation='softmax')
  ])
  return model

model = build_model()
```


```python
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
```


```python
history = model.fit(train_data,validation_data=valid_data,epochs=50)
```

    Epoch 1/50
    338/338 [==============================] - 15s 21ms/step - loss: 1.6741 - acc: 0.4277 - val_loss: 1.6587 - val_acc: 0.4246
    Epoch 2/50
    338/338 [==============================] - 4s 13ms/step - loss: 1.2652 - acc: 0.5557 - val_loss: 1.0137 - val_acc: 0.6717
    Epoch 3/50
    338/338 [==============================] - 4s 12ms/step - loss: 1.0971 - acc: 0.6121 - val_loss: 0.8198 - val_acc: 0.7293
    Epoch 4/50
    338/338 [==============================] - 4s 13ms/step - loss: 0.9839 - acc: 0.6553 - val_loss: 0.7896 - val_acc: 0.7341
    Epoch 5/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.9075 - acc: 0.6840 - val_loss: 0.6642 - val_acc: 0.7793
    Epoch 6/50
    338/338 [==============================] - 4s 13ms/step - loss: 0.8436 - acc: 0.7126 - val_loss: 0.6745 - val_acc: 0.7681
    Epoch 7/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.7838 - acc: 0.7348 - val_loss: 0.6896 - val_acc: 0.7776
    Epoch 8/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.7401 - acc: 0.7527 - val_loss: 0.5589 - val_acc: 0.8259
    Epoch 9/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.6741 - acc: 0.7751 - val_loss: 0.5548 - val_acc: 0.8209
    Epoch 10/50
    338/338 [==============================] - 4s 13ms/step - loss: 0.6382 - acc: 0.7925 - val_loss: 0.6320 - val_acc: 0.7776
    Epoch 11/50
    338/338 [==============================] - 5s 13ms/step - loss: 0.6167 - acc: 0.7974 - val_loss: 0.5035 - val_acc: 0.8387
    Epoch 12/50
    338/338 [==============================] - 7s 21ms/step - loss: 0.5593 - acc: 0.8186 - val_loss: 0.5071 - val_acc: 0.8417
    Epoch 13/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.5260 - acc: 0.8299 - val_loss: 0.4805 - val_acc: 0.8422
    Epoch 14/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.4911 - acc: 0.8436 - val_loss: 0.4991 - val_acc: 0.8413
    Epoch 15/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.4683 - acc: 0.8520 - val_loss: 0.4907 - val_acc: 0.8444
    Epoch 16/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.4520 - acc: 0.8603 - val_loss: 0.4767 - val_acc: 0.8502
    Epoch 17/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.4285 - acc: 0.8621 - val_loss: 0.4463 - val_acc: 0.8641
    Epoch 18/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.4050 - acc: 0.8739 - val_loss: 0.4643 - val_acc: 0.8591
    Epoch 19/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.3843 - acc: 0.8810 - val_loss: 0.4821 - val_acc: 0.8557
    Epoch 20/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.3721 - acc: 0.8828 - val_loss: 0.4706 - val_acc: 0.8626
    Epoch 21/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.3695 - acc: 0.8828 - val_loss: 0.4900 - val_acc: 0.8522
    Epoch 22/50
    338/338 [==============================] - 5s 13ms/step - loss: 0.3443 - acc: 0.8944 - val_loss: 0.4878 - val_acc: 0.8659
    Epoch 23/50
    338/338 [==============================] - 5s 13ms/step - loss: 0.3541 - acc: 0.8919 - val_loss: 0.4893 - val_acc: 0.8620
    Epoch 24/50
    338/338 [==============================] - 5s 13ms/step - loss: 0.3299 - acc: 0.8988 - val_loss: 0.4672 - val_acc: 0.8715
    Epoch 25/50
    338/338 [==============================] - 4s 13ms/step - loss: 0.3144 - acc: 0.9051 - val_loss: 0.4442 - val_acc: 0.8724
    Epoch 26/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.2984 - acc: 0.9092 - val_loss: 0.5007 - val_acc: 0.8580
    Epoch 27/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.2931 - acc: 0.9133 - val_loss: 0.5533 - val_acc: 0.8526
    Epoch 28/50
    338/338 [==============================] - 4s 13ms/step - loss: 0.2945 - acc: 0.9103 - val_loss: 0.5272 - val_acc: 0.8589
    Epoch 29/50
    338/338 [==============================] - 4s 13ms/step - loss: 0.2772 - acc: 0.9171 - val_loss: 0.5181 - val_acc: 0.8683
    Epoch 30/50
    338/338 [==============================] - 4s 13ms/step - loss: 0.2649 - acc: 0.9208 - val_loss: 0.5178 - val_acc: 0.8669
    Epoch 31/50
    338/338 [==============================] - 4s 13ms/step - loss: 0.2789 - acc: 0.9175 - val_loss: 0.5736 - val_acc: 0.8593
    Epoch 32/50
    338/338 [==============================] - 4s 13ms/step - loss: 0.2570 - acc: 0.9216 - val_loss: 0.5390 - val_acc: 0.8693
    Epoch 33/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.2577 - acc: 0.9216 - val_loss: 0.5505 - val_acc: 0.8702
    Epoch 34/50
    338/338 [==============================] - 4s 13ms/step - loss: 0.2505 - acc: 0.9264 - val_loss: 0.5240 - val_acc: 0.8646
    Epoch 35/50
    338/338 [==============================] - 4s 13ms/step - loss: 0.2435 - acc: 0.9301 - val_loss: 0.5341 - val_acc: 0.8635
    Epoch 36/50
    338/338 [==============================] - 5s 13ms/step - loss: 0.2455 - acc: 0.9287 - val_loss: 0.5612 - val_acc: 0.8687
    Epoch 37/50
    338/338 [==============================] - 5s 13ms/step - loss: 0.2291 - acc: 0.9320 - val_loss: 0.6070 - val_acc: 0.8689
    Epoch 38/50
    338/338 [==============================] - 4s 13ms/step - loss: 0.2416 - acc: 0.9293 - val_loss: 0.5367 - val_acc: 0.8661
    Epoch 39/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.2395 - acc: 0.9296 - val_loss: 0.5004 - val_acc: 0.8700
    Epoch 40/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.2215 - acc: 0.9349 - val_loss: 0.5522 - val_acc: 0.8667
    Epoch 41/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.2334 - acc: 0.9313 - val_loss: 0.4850 - val_acc: 0.8752
    Epoch 42/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.2340 - acc: 0.9324 - val_loss: 0.5437 - val_acc: 0.8706
    Epoch 43/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.2270 - acc: 0.9328 - val_loss: 0.5510 - val_acc: 0.8676
    Epoch 44/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.2100 - acc: 0.9392 - val_loss: 0.5819 - val_acc: 0.8669
    Epoch 45/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.2146 - acc: 0.9352 - val_loss: 0.5407 - val_acc: 0.8702
    Epoch 46/50
    338/338 [==============================] - 4s 12ms/step - loss: 0.2069 - acc: 0.9399 - val_loss: 0.6184 - val_acc: 0.8548
    Epoch 47/50
    338/338 [==============================] - 4s 13ms/step - loss: 0.2100 - acc: 0.9396 - val_loss: 0.5637 - val_acc: 0.8785
    Epoch 48/50
    338/338 [==============================] - 5s 14ms/step - loss: 0.2103 - acc: 0.9394 - val_loss: 0.5665 - val_acc: 0.8735
    Epoch 49/50
    338/338 [==============================] - 5s 14ms/step - loss: 0.1965 - acc: 0.9435 - val_loss: 0.5312 - val_acc: 0.8728
    Epoch 50/50
    338/338 [==============================] - 4s 13ms/step - loss: 0.1959 - acc: 0.9439 - val_loss: 0.5340 - val_acc: 0.8785
    


```python
def plot_loss_acc(history,epoch):
  loss,val_loss = history.history['loss'],history.history['val_loss']
  acc,val_acc = history.history['acc'],history.history['val_acc']

  fig, axes = plt.subplots(1,2,figsize=(12,4))

  axes[0].plot(range(1, epoch+1), loss, label='train_loss')
  axes[0].plot(range(1, epoch+1), val_loss, label='valid_loss')
  axes[0].legend(loc='best')
  axes[0].set_title('Loss')

  axes[1].plot(range(1, epoch+1), acc, label='train_acc')
  axes[1].plot(range(1, epoch+1), val_acc, label='valid_acc')
  axes[1].legend(loc='best')
  axes[1].set_title('Acc')

  plt.show()

plot_loss_acc(history,50)
```


    
![png](16_cnn_euroset_files/16_cnn_euroset_14_0.png)
    



```python
image_batch , label_batch = next(iter(train_data.take(1)))
image_batch.shape,label_batch.shape
```




    (TensorShape([64, 64, 64, 3]), TensorShape([64]))




```python
image = image_batch[0]
label = label_batch[0].numpy()
image,label
```


```python
plt.imshow(image)
plt.title(info.features['label'].int2str(label))
```




    Text(0.5, 1.0, 'Industrial')




    
![png](16_cnn_euroset_files/16_cnn_euroset_17_1.png)
    



```python
def plot_augmentation(original,augmented):
  fig,axes = plt.subplots(1,2,figsize=(12,4))

  axes[0].imshow(original)
  axes[0].set_title('Original')

  axes[1].imshow(augmented)
  axes[1].set_title('Augmented')

plt.show()
```


```python
plot_augmentation(image,tf.image.flip_left_right(image))
```


    
![png](16_cnn_euroset_files/16_cnn_euroset_19_0.png)
    



```python
plot_augmentation(image,tf.image.flip_up_down(image))
```


    
![png](16_cnn_euroset_files/16_cnn_euroset_20_0.png)
    



```python
plot_augmentation(image,tf.image.rot90(image))
```


    
![png](16_cnn_euroset_files/16_cnn_euroset_21_0.png)
    



```python
plot_augmentation(image,tf.image.transpose(image))
```


    
![png](16_cnn_euroset_files/16_cnn_euroset_22_0.png)
    



```python
plot_augmentation(image,tf.image.central_crop(image,central_fraction=0.6))
```


    
![png](16_cnn_euroset_files/16_cnn_euroset_23_0.png)
    



```python
plot_augmentation(image,tf.image.resize_with_crop_or_pad(image, 64+20, 64+20))
```


    
![png](16_cnn_euroset_files/16_cnn_euroset_24_0.png)
    



```python
plot_augmentation(image,tf.image.adjust_brightness(image,0.3))
```

    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    
![png](16_cnn_euroset_files/16_cnn_euroset_25_1.png)
    



```python
plot_augmentation(image,tf.image.adjust_saturation(image,0.5))
```


    
![png](16_cnn_euroset_files/16_cnn_euroset_26_0.png)
    



```python
plot_augmentation(image,tf.image.adjust_contrast(image,2))
```

    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    
![png](16_cnn_euroset_files/16_cnn_euroset_27_1.png)
    



```python
plot_augmentation(image,tf.image.random_crop(image,size=[64,64,3]))
```


    
![png](16_cnn_euroset_files/16_cnn_euroset_28_0.png)
    



```python
batch_size=64
buffer_size = 1000

def preprocess_data(image,label):
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_flip_up_down(image)
  image = tf.image.random_brightness(image, max_delta=0.3)
  image = tf.image.random_contrast(image,0.5,1.5)
  image = tf.cast(image,tf.float32)/255.
  return image,label

train_aug = train_ds.map(preprocess_data,num_parallel_calls=tf.data.AUTOTUNE)
valid_aug = valid_ds.map(preprocess_data,num_parallel_calls=tf.data.AUTOTUNE)

train_aug = train_aug.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
valid_aug = valid_aug.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
```


```python
image_batch , label_batch = next(iter(train_aug.take(1)))
image_batch.shape,label_batch.shape

image = image_batch[0]
label = label_batch[0].numpy()
# image,label

plt.imshow(image)
plt.title(info.features['label'].int2str(label))
```




    Text(0.5, 1.0, 'River')




    
![png](16_cnn_euroset_files/16_cnn_euroset_30_1.png)
    



```python
aug_model = build_model()
aug_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
aug_history = aug_model.fit(train_aug,validation_data=valid_aug,epochs=50)
plot_loss_acc(aug_history,50)
```

    Epoch 1/50
    338/338 [==============================] - 8s 22ms/step - loss: 2.0178 - acc: 0.2523 - val_loss: 1.7844 - val_acc: 0.3406
    Epoch 2/50
    338/338 [==============================] - 9s 25ms/step - loss: 1.7576 - acc: 0.3144 - val_loss: 1.5007 - val_acc: 0.4502
    Epoch 3/50
    338/338 [==============================] - 10s 27ms/step - loss: 1.6351 - acc: 0.3576 - val_loss: 1.4204 - val_acc: 0.4570
    Epoch 4/50
    338/338 [==============================] - 9s 26ms/step - loss: 1.5682 - acc: 0.3948 - val_loss: 1.3530 - val_acc: 0.5028
    Epoch 5/50
    338/338 [==============================] - 10s 29ms/step - loss: 1.4914 - acc: 0.4258 - val_loss: 1.3627 - val_acc: 0.4920
    Epoch 6/50
    338/338 [==============================] - 9s 27ms/step - loss: 1.4411 - acc: 0.4500 - val_loss: 1.2386 - val_acc: 0.5441
    Epoch 7/50
    338/338 [==============================] - 14s 37ms/step - loss: 1.3866 - acc: 0.4750 - val_loss: 1.2053 - val_acc: 0.5496
    Epoch 8/50
    338/338 [==============================] - 11s 30ms/step - loss: 1.3614 - acc: 0.4907 - val_loss: 1.2495 - val_acc: 0.5307
    Epoch 9/50
    338/338 [==============================] - 7s 21ms/step - loss: 1.3008 - acc: 0.5150 - val_loss: 1.0876 - val_acc: 0.6122
    Epoch 10/50
    338/338 [==============================] - 7s 20ms/step - loss: 1.2607 - acc: 0.5355 - val_loss: 1.1187 - val_acc: 0.6009
    Epoch 11/50
    338/338 [==============================] - 7s 21ms/step - loss: 1.2275 - acc: 0.5547 - val_loss: 1.1229 - val_acc: 0.6198
    Epoch 12/50
    338/338 [==============================] - 7s 21ms/step - loss: 1.1810 - acc: 0.5742 - val_loss: 1.0530 - val_acc: 0.6278
    Epoch 13/50
    338/338 [==============================] - 7s 21ms/step - loss: 1.1703 - acc: 0.5811 - val_loss: 0.9749 - val_acc: 0.6596
    Epoch 14/50
    338/338 [==============================] - 7s 21ms/step - loss: 1.1395 - acc: 0.5875 - val_loss: 0.9768 - val_acc: 0.6567
    Epoch 15/50
    338/338 [==============================] - 7s 21ms/step - loss: 1.1247 - acc: 0.5974 - val_loss: 0.9234 - val_acc: 0.6920
    Epoch 16/50
    338/338 [==============================] - 7s 21ms/step - loss: 1.0914 - acc: 0.6097 - val_loss: 0.9391 - val_acc: 0.6863
    Epoch 17/50
    338/338 [==============================] - 8s 22ms/step - loss: 1.0521 - acc: 0.6246 - val_loss: 0.9591 - val_acc: 0.6774
    Epoch 18/50
    338/338 [==============================] - 7s 20ms/step - loss: 1.0502 - acc: 0.6262 - val_loss: 0.8416 - val_acc: 0.7209
    Epoch 19/50
    338/338 [==============================] - 7s 21ms/step - loss: 1.0072 - acc: 0.6455 - val_loss: 0.8898 - val_acc: 0.6863
    Epoch 20/50
    338/338 [==============================] - 8s 22ms/step - loss: 0.9811 - acc: 0.6510 - val_loss: 1.0240 - val_acc: 0.6128
    Epoch 21/50
    338/338 [==============================] - 7s 22ms/step - loss: 0.9620 - acc: 0.6562 - val_loss: 0.8306 - val_acc: 0.7126
    Epoch 22/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.9450 - acc: 0.6641 - val_loss: 0.8159 - val_acc: 0.7065
    Epoch 23/50
    338/338 [==============================] - 7s 21ms/step - loss: 0.9411 - acc: 0.6643 - val_loss: 0.7896 - val_acc: 0.7244
    Epoch 24/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.9073 - acc: 0.6770 - val_loss: 0.7280 - val_acc: 0.7520
    Epoch 25/50
    338/338 [==============================] - 13s 37ms/step - loss: 0.8805 - acc: 0.6929 - val_loss: 0.7174 - val_acc: 0.7570
    Epoch 26/50
    338/338 [==============================] - 9s 25ms/step - loss: 0.8586 - acc: 0.7000 - val_loss: 0.7128 - val_acc: 0.7550
    Epoch 27/50
    338/338 [==============================] - 7s 21ms/step - loss: 0.8253 - acc: 0.7198 - val_loss: 0.7023 - val_acc: 0.7557
    Epoch 28/50
    338/338 [==============================] - 7s 21ms/step - loss: 0.8063 - acc: 0.7253 - val_loss: 0.6620 - val_acc: 0.7759
    Epoch 29/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.7852 - acc: 0.7348 - val_loss: 0.6701 - val_acc: 0.7776
    Epoch 30/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.7855 - acc: 0.7340 - val_loss: 0.6574 - val_acc: 0.7804
    Epoch 31/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.7712 - acc: 0.7422 - val_loss: 0.6436 - val_acc: 0.7815
    Epoch 32/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.7501 - acc: 0.7433 - val_loss: 0.6716 - val_acc: 0.7765
    Epoch 33/50
    338/338 [==============================] - 7s 21ms/step - loss: 0.7403 - acc: 0.7502 - val_loss: 0.6855 - val_acc: 0.7472
    Epoch 34/50
    338/338 [==============================] - 7s 21ms/step - loss: 0.7211 - acc: 0.7545 - val_loss: 0.6403 - val_acc: 0.7802
    Epoch 35/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.7333 - acc: 0.7526 - val_loss: 0.5883 - val_acc: 0.8024
    Epoch 36/50
    338/338 [==============================] - 7s 21ms/step - loss: 0.7056 - acc: 0.7643 - val_loss: 0.5776 - val_acc: 0.8080
    Epoch 37/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.6891 - acc: 0.7691 - val_loss: 0.6179 - val_acc: 0.7922
    Epoch 38/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.6733 - acc: 0.7769 - val_loss: 0.6247 - val_acc: 0.7844
    Epoch 39/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.6603 - acc: 0.7804 - val_loss: 0.5868 - val_acc: 0.8048
    Epoch 40/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.6388 - acc: 0.7919 - val_loss: 0.5817 - val_acc: 0.7917
    Epoch 41/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.6285 - acc: 0.7901 - val_loss: 0.5749 - val_acc: 0.8078
    Epoch 42/50
    338/338 [==============================] - 8s 24ms/step - loss: 0.6256 - acc: 0.7931 - val_loss: 0.5904 - val_acc: 0.7943
    Epoch 43/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.6404 - acc: 0.7918 - val_loss: 0.5662 - val_acc: 0.8091
    Epoch 44/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.6095 - acc: 0.7962 - val_loss: 0.5323 - val_acc: 0.8172
    Epoch 45/50
    338/338 [==============================] - 8s 23ms/step - loss: 0.6238 - acc: 0.7962 - val_loss: 0.5385 - val_acc: 0.8180
    Epoch 46/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.6140 - acc: 0.7969 - val_loss: 0.5863 - val_acc: 0.8033
    Epoch 47/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.6113 - acc: 0.7972 - val_loss: 0.5587 - val_acc: 0.8137
    Epoch 48/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.6009 - acc: 0.8032 - val_loss: 0.5820 - val_acc: 0.7919
    Epoch 49/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.6050 - acc: 0.8011 - val_loss: 0.5275 - val_acc: 0.8196
    Epoch 50/50
    338/338 [==============================] - 7s 20ms/step - loss: 0.5799 - acc: 0.8091 - val_loss: 0.5147 - val_acc: 0.8265
    


    
![png](16_cnn_euroset_files/16_cnn_euroset_31_1.png)
    



```python
from keras.applications import ResNet50V2
from keras.utils import plot_model

pre_trained_base = ResNet50V2(include_top=False,input_shape=[64,64,3])
pre_trained_base.trainable = False
plot_model(pre_trained_base,show_shapes=True,show_layer_names=True)
```


    Output hidden; open in https://colab.research.google.com to view.



```python
def build_transfer_model():
  model = Sequential([
      pre_trained_base,

      Flatten(),
      Dense(128,activation='relu'),
      Dropout(0.3),

      Dense(64,activation='relu'),
      Dropout(0.3),
      Dense(num_classes,activation='softmax')
  ])
  return model
```


```python
t_model = build_transfer_model()
t_model.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     resnet50v2 (Functional)     (None, 2, 2, 2048)        23564800  
                                                                     
     flatten_3 (Flatten)         (None, 8192)              0         
                                                                     
     dense_9 (Dense)             (None, 128)               1048704   
                                                                     
     dropout_6 (Dropout)         (None, 128)               0         
                                                                     
     dense_10 (Dense)            (None, 64)                8256      
                                                                     
     dropout_7 (Dropout)         (None, 64)                0         
                                                                     
     dense_11 (Dense)            (None, 10)                650       
                                                                     
    =================================================================
    Total params: 24,622,410
    Trainable params: 1,057,610
    Non-trainable params: 23,564,800
    _________________________________________________________________
    


```python
t_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
t_history = t_model.fit(train_aug,validation_data=valid_aug,epochs=50)
plot_loss_acc(aug_history,50)
```

    Epoch 1/50
    338/338 [==============================] - 17s 35ms/step - loss: 1.1214 - acc: 0.6397 - val_loss: 0.6685 - val_acc: 0.7769
    Epoch 2/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.7960 - acc: 0.7432 - val_loss: 0.5797 - val_acc: 0.8081
    Epoch 3/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.7125 - acc: 0.7693 - val_loss: 0.5438 - val_acc: 0.8154
    Epoch 4/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.6593 - acc: 0.7852 - val_loss: 0.5251 - val_acc: 0.8220
    Epoch 5/50
    338/338 [==============================] - 11s 33ms/step - loss: 0.6299 - acc: 0.7945 - val_loss: 0.5116 - val_acc: 0.8239
    Epoch 6/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.6072 - acc: 0.8009 - val_loss: 0.5073 - val_acc: 0.8309
    Epoch 7/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.5837 - acc: 0.8077 - val_loss: 0.4921 - val_acc: 0.8319
    Epoch 8/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.5706 - acc: 0.8150 - val_loss: 0.4921 - val_acc: 0.8311
    Epoch 9/50
    338/338 [==============================] - 11s 30ms/step - loss: 0.5663 - acc: 0.8159 - val_loss: 0.4817 - val_acc: 0.8333
    Epoch 10/50
    338/338 [==============================] - 11s 32ms/step - loss: 0.5448 - acc: 0.8198 - val_loss: 0.4681 - val_acc: 0.8404
    Epoch 11/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.5502 - acc: 0.8199 - val_loss: 0.4778 - val_acc: 0.8337
    Epoch 12/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.5330 - acc: 0.8244 - val_loss: 0.4686 - val_acc: 0.8394
    Epoch 13/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.5380 - acc: 0.8263 - val_loss: 0.4609 - val_acc: 0.8409
    Epoch 14/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.5233 - acc: 0.8275 - val_loss: 0.4610 - val_acc: 0.8393
    Epoch 15/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.5119 - acc: 0.8317 - val_loss: 0.4750 - val_acc: 0.8346
    Epoch 16/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.5037 - acc: 0.8365 - val_loss: 0.4476 - val_acc: 0.8456
    Epoch 17/50
    338/338 [==============================] - 11s 32ms/step - loss: 0.5093 - acc: 0.8344 - val_loss: 0.4464 - val_acc: 0.8472
    Epoch 18/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.4918 - acc: 0.8342 - val_loss: 0.4441 - val_acc: 0.8426
    Epoch 19/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.4918 - acc: 0.8366 - val_loss: 0.4452 - val_acc: 0.8446
    Epoch 20/50
    338/338 [==============================] - 11s 32ms/step - loss: 0.4850 - acc: 0.8397 - val_loss: 0.4533 - val_acc: 0.8450
    Epoch 21/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.4799 - acc: 0.8404 - val_loss: 0.4454 - val_acc: 0.8456
    Epoch 22/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.4811 - acc: 0.8405 - val_loss: 0.4468 - val_acc: 0.8459
    Epoch 23/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.4638 - acc: 0.8475 - val_loss: 0.4322 - val_acc: 0.8509
    Epoch 24/50
    338/338 [==============================] - 11s 30ms/step - loss: 0.4746 - acc: 0.8429 - val_loss: 0.4500 - val_acc: 0.8446
    Epoch 25/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4577 - acc: 0.8480 - val_loss: 0.4506 - val_acc: 0.8454
    Epoch 26/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4587 - acc: 0.8478 - val_loss: 0.4335 - val_acc: 0.8494
    Epoch 27/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4671 - acc: 0.8444 - val_loss: 0.4322 - val_acc: 0.8489
    Epoch 28/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4478 - acc: 0.8515 - val_loss: 0.4288 - val_acc: 0.8498
    Epoch 29/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4483 - acc: 0.8504 - val_loss: 0.4377 - val_acc: 0.8493
    Epoch 30/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4539 - acc: 0.8494 - val_loss: 0.4423 - val_acc: 0.8456
    Epoch 31/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4507 - acc: 0.8515 - val_loss: 0.4307 - val_acc: 0.8467
    Epoch 32/50
    338/338 [==============================] - 11s 33ms/step - loss: 0.4405 - acc: 0.8555 - val_loss: 0.4313 - val_acc: 0.8500
    Epoch 33/50
    338/338 [==============================] - 11s 30ms/step - loss: 0.4362 - acc: 0.8514 - val_loss: 0.4377 - val_acc: 0.8478
    Epoch 34/50
    338/338 [==============================] - 11s 30ms/step - loss: 0.4418 - acc: 0.8518 - val_loss: 0.4219 - val_acc: 0.8498
    Epoch 35/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4354 - acc: 0.8555 - val_loss: 0.4308 - val_acc: 0.8476
    Epoch 36/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.4339 - acc: 0.8562 - val_loss: 0.4297 - val_acc: 0.8500
    Epoch 37/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4287 - acc: 0.8547 - val_loss: 0.4361 - val_acc: 0.8489
    Epoch 38/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4243 - acc: 0.8592 - val_loss: 0.4272 - val_acc: 0.8531
    Epoch 39/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4180 - acc: 0.8630 - val_loss: 0.4363 - val_acc: 0.8494
    Epoch 40/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4253 - acc: 0.8597 - val_loss: 0.4270 - val_acc: 0.8531
    Epoch 41/50
    338/338 [==============================] - 11s 30ms/step - loss: 0.4244 - acc: 0.8572 - val_loss: 0.4282 - val_acc: 0.8539
    Epoch 42/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4194 - acc: 0.8571 - val_loss: 0.4226 - val_acc: 0.8539
    Epoch 43/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4162 - acc: 0.8591 - val_loss: 0.4315 - val_acc: 0.8526
    Epoch 44/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4246 - acc: 0.8562 - val_loss: 0.4306 - val_acc: 0.8543
    Epoch 45/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.4131 - acc: 0.8601 - val_loss: 0.4307 - val_acc: 0.8522
    Epoch 46/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.4130 - acc: 0.8621 - val_loss: 0.4287 - val_acc: 0.8544
    Epoch 47/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.4015 - acc: 0.8650 - val_loss: 0.4271 - val_acc: 0.8578
    Epoch 48/50
    338/338 [==============================] - 10s 30ms/step - loss: 0.4098 - acc: 0.8630 - val_loss: 0.4287 - val_acc: 0.8544
    Epoch 49/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.4032 - acc: 0.8671 - val_loss: 0.4170 - val_acc: 0.8631
    Epoch 50/50
    338/338 [==============================] - 11s 31ms/step - loss: 0.4009 - acc: 0.8635 - val_loss: 0.4285 - val_acc: 0.8539
    


    
![png](16_cnn_euroset_files/16_cnn_euroset_35_1.png)
    



```python
pred = t_model.predict(image.numpy().reshape(-1,64,64,3))
pred
```

    1/1 [==============================] - 2s 2s/step
    




    array([[6.0538650e-03, 2.8989831e-05, 8.7860419e-04, 3.2317329e-02,
            1.4384495e-05, 2.2318615e-02, 2.9983968e-04, 7.3251408e-07,
            9.3794680e-01, 1.4082357e-04]], dtype=float32)




```python

```
