## 컨볼루션 신경망(Convolution Neural Network)
### 일단 적용해보기


```python
from keras.datasets import fashion_mnist
```


```python
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
    29515/29515 [==============================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
    26421880/26421880 [==============================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
    5148/5148 [==============================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
    4422102/4422102 [==============================] - 0s 0us/step
    


```python
import matplotlib.pyplot as plt
import numpy as np
```


```python
x_train.shape
```




    (60000, 28, 28)




```python
np.random.seed(777)
class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
sample_size = 9
# 0~59999에서 무작위로 정수값 추출
random_idx = np.random.randint(60000, size = sample_size)
x_train = np.reshape(x_train/255,(-1,28,28,1)) # 맨 뒤에 흑백의 컬러값으로 1을 넣어줌 ex) grayscale=3
x_test = np.reshape(x_test/255,(-1,28,28,1))
```


```python
x_train.shape
```




    (60000, 28, 28, 1)




```python
x_train.min(),x_train.max()
```




    (0.0, 1.0)




```python
y_train[0]
```




    9




```python
from keras.utils import to_categorical
```


```python
# 각 데이터의 레이블을 범주형 형태로 변경
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```


```python
y_train[0]
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.], dtype=float32)




```python
from sklearn.model_selection import train_test_split
```


```python
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.3,random_state=777)
```


```python
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
```


```python
# 모델 구성하기
model = Sequential([
    Conv2D(filters = 16, kernel_size = 3, strides = (1,1), padding = 'same', activation='relu', input_shape = (28, 28, 1)),
    MaxPool2D(pool_size = (2,2), strides=2, padding='same'),
    # 필터를 늘려서 층 수를 추가로 적용
    Conv2D(filters = 32, kernel_size = 3, strides = (1,1), padding = 'same', activation='relu'),
    MaxPool2D(pool_size = (2,2), strides=2, padding='same'),
    Conv2D(filters = 64, kernel_size = 3, strides = (1,1), padding = 'same', activation='relu'),
    MaxPool2D(pool_size = (2,2), strides=2, padding='same'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 학습시키기
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 28, 28, 16)        160       
                                                                     
     max_pooling2d (MaxPooling2D  (None, 14, 14, 16)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 14, 14, 32)        4640      
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 7, 7, 32)         0         
     2D)                                                             
                                                                     
     conv2d_2 (Conv2D)           (None, 7, 7, 64)          18496     
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 4, 4, 64)         0         
     2D)                                                             
                                                                     
     flatten (Flatten)           (None, 1024)              0         
                                                                     
     dense (Dense)               (None, 64)                65600     
                                                                     
     dense_1 (Dense)             (None, 10)                650       
                                                                     
    =================================================================
    Total params: 89,546
    Trainable params: 89,546
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.fit(x_train,y_train,epochs=30,batch_size=32,validation_data=(x_val,y_val))
```

    Epoch 1/30
    1313/1313 [==============================] - 49s 38ms/step - loss: 0.4646 - acc: 0.8325 - val_loss: 0.3444 - val_acc: 0.8775
    Epoch 2/30
    1313/1313 [==============================] - 53s 40ms/step - loss: 0.3265 - acc: 0.8824 - val_loss: 0.3151 - val_acc: 0.8857
    Epoch 3/30
    1313/1313 [==============================] - 51s 39ms/step - loss: 0.2817 - acc: 0.8967 - val_loss: 0.3020 - val_acc: 0.8896
    Epoch 4/30
    1313/1313 [==============================] - 49s 37ms/step - loss: 0.2533 - acc: 0.9065 - val_loss: 0.2622 - val_acc: 0.9063
    Epoch 5/30
    1313/1313 [==============================] - 51s 39ms/step - loss: 0.2303 - acc: 0.9160 - val_loss: 0.2696 - val_acc: 0.9051
    Epoch 6/30
    1313/1313 [==============================] - 51s 39ms/step - loss: 0.2089 - acc: 0.9220 - val_loss: 0.2663 - val_acc: 0.9054
    Epoch 7/30
    1313/1313 [==============================] - 51s 39ms/step - loss: 0.1898 - acc: 0.9294 - val_loss: 0.2416 - val_acc: 0.9139
    Epoch 8/30
    1313/1313 [==============================] - 50s 38ms/step - loss: 0.1726 - acc: 0.9360 - val_loss: 0.2361 - val_acc: 0.9159
    Epoch 9/30
    1313/1313 [==============================] - 52s 40ms/step - loss: 0.1581 - acc: 0.9403 - val_loss: 0.2517 - val_acc: 0.9132
    Epoch 10/30
    1313/1313 [==============================] - 52s 40ms/step - loss: 0.1436 - acc: 0.9455 - val_loss: 0.2660 - val_acc: 0.9094
    Epoch 11/30
    1313/1313 [==============================] - 46s 35ms/step - loss: 0.1295 - acc: 0.9522 - val_loss: 0.2897 - val_acc: 0.9063
    Epoch 12/30
    1313/1313 [==============================] - 46s 35ms/step - loss: 0.1215 - acc: 0.9535 - val_loss: 0.2654 - val_acc: 0.9115
    Epoch 13/30
    1313/1313 [==============================] - 46s 35ms/step - loss: 0.1067 - acc: 0.9599 - val_loss: 0.2767 - val_acc: 0.9121
    Epoch 14/30
    1313/1313 [==============================] - 46s 35ms/step - loss: 0.0977 - acc: 0.9629 - val_loss: 0.2836 - val_acc: 0.9186
    Epoch 15/30
    1313/1313 [==============================] - 47s 36ms/step - loss: 0.0878 - acc: 0.9667 - val_loss: 0.2877 - val_acc: 0.9169
    Epoch 16/30
    1313/1313 [==============================] - 50s 38ms/step - loss: 0.0777 - acc: 0.9700 - val_loss: 0.3274 - val_acc: 0.9140
    Epoch 17/30
    1313/1313 [==============================] - 46s 35ms/step - loss: 0.0722 - acc: 0.9724 - val_loss: 0.3582 - val_acc: 0.9157
    Epoch 18/30
    1313/1313 [==============================] - 46s 35ms/step - loss: 0.0643 - acc: 0.9757 - val_loss: 0.3668 - val_acc: 0.9085
    Epoch 19/30
    1313/1313 [==============================] - 46s 35ms/step - loss: 0.0641 - acc: 0.9761 - val_loss: 0.3634 - val_acc: 0.9168
    Epoch 20/30
    1313/1313 [==============================] - 46s 35ms/step - loss: 0.0601 - acc: 0.9771 - val_loss: 0.3814 - val_acc: 0.9133
    Epoch 21/30
    1313/1313 [==============================] - 50s 38ms/step - loss: 0.0512 - acc: 0.9806 - val_loss: 0.3971 - val_acc: 0.9107
    Epoch 22/30
    1313/1313 [==============================] - 47s 36ms/step - loss: 0.0495 - acc: 0.9813 - val_loss: 0.4113 - val_acc: 0.9106
    Epoch 23/30
    1313/1313 [==============================] - 48s 36ms/step - loss: 0.0474 - acc: 0.9828 - val_loss: 0.4283 - val_acc: 0.9175
    Epoch 24/30
    1313/1313 [==============================] - 48s 37ms/step - loss: 0.0473 - acc: 0.9818 - val_loss: 0.4396 - val_acc: 0.9102
    Epoch 25/30
    1313/1313 [==============================] - 47s 36ms/step - loss: 0.0386 - acc: 0.9859 - val_loss: 0.4718 - val_acc: 0.9119
    Epoch 26/30
    1313/1313 [==============================] - 46s 35ms/step - loss: 0.0363 - acc: 0.9862 - val_loss: 0.4975 - val_acc: 0.9182
    Epoch 27/30
    1313/1313 [==============================] - 50s 38ms/step - loss: 0.0409 - acc: 0.9852 - val_loss: 0.4568 - val_acc: 0.9130
    Epoch 28/30
    1313/1313 [==============================] - 57s 43ms/step - loss: 0.0371 - acc: 0.9858 - val_loss: 0.5332 - val_acc: 0.9109
    Epoch 29/30
    1313/1313 [==============================] - 51s 39ms/step - loss: 0.0347 - acc: 0.9874 - val_loss: 0.5286 - val_acc: 0.9118
    Epoch 30/30
    1313/1313 [==============================] - 46s 35ms/step - loss: 0.0365 - acc: 0.9864 - val_loss: 0.5274 - val_acc: 0.9129
    




    <keras.callbacks.History at 0x7f65801b0f70>




```python
from keras.utils import plot_model
```


```python
plot_model(model, './model.png',show_shapes=True)
```




    
![png](07_cnn_fashion_mnist_files/07_cnn_fashion_mnist_18_0.png)
    




```python
# 테스트 결과
pred = model.predict(x_test)
```

    313/313 [==============================] - 4s 11ms/step
    


```python
# 테스트 결과값 확인
pred[0]
```




    array([8.9170865e-14, 4.2583481e-25, 9.5145649e-20, 1.6950283e-20,
           3.7162090e-18, 3.9351281e-07, 5.7551694e-23, 2.8031328e-09,
           7.0376129e-18, 9.9999958e-01], dtype=float32)




```python
np.round(pred[0],2)
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.], dtype=float32)




```python
class_name[np.argmax(np.round(pred[0],2))]
```




    'Ankle boot'




```python
x_test.shape
```




    (10000, 28, 28, 1)




```python
# 출력
plt.imshow(x_test[0].reshape((28,28)),cmap='gray') # (28,28,1)인 3차원이므로 reshape로 형태변환을 해줘야 함
```




    <matplotlib.image.AxesImage at 0x7f657f8499a0>




    
![png](07_cnn_fashion_mnist_files/07_cnn_fashion_mnist_24_1.png)
    



```python
# 모델 저장. 경로 지정 후 이름을 추가로 써줘야 함
model.save_weights('/content/drive/MyDrive/Colab Notebooks/sesac_deeplearning/model/07_cnn_fashion_mnist/model')
```


```python
# 신규 모델 구성
model1 = Sequential([
    Conv2D(filters = 16, kernel_size = 3, strides = (1,1), padding = 'same', activation='relu', input_shape = (28, 28, 1)),
    MaxPool2D(pool_size = (2,2), strides=2, padding='same'),
    Conv2D(filters = 32, kernel_size = 3, strides = (1,1), padding = 'same', activation='relu'),
    MaxPool2D(pool_size = (2,2), strides=2, padding='same'),
    Conv2D(filters = 64, kernel_size = 3, strides = (1,1), padding = 'same', activation='relu'),
    MaxPool2D(pool_size = (2,2), strides=2, padding='same'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```


```python
model1.load_weights('/content/drive/MyDrive/Colab Notebooks/sesac_deeplearning/model/07_cnn_fashion_mnist/model')
```




    <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x7f657f35c550>




```python
pred = model1.predict(x_test)
```

    313/313 [==============================] - 3s 10ms/step
    


```python
# 신규 모델 결과 확인
np.round(pred[0],2)
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.], dtype=float32)


