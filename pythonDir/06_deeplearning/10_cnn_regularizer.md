```python
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 채널별로 평균과 표준편차 구하기기
x_mean = np.mean(x_train,axis=(0,1,2))
x_std = np.std(x_train, axis = (0,1,2))

x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.3,random_state=777)
```

    Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    170498071/170498071 [==============================] - 8s 0us/step
    


```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam
from keras.regularizers import l2

model = Sequential([
    Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=(32,32,3)),
    Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'),
    MaxPool2D(pool_size=(2,2),strides=2,padding='same'),

    Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'),
    Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'),
    MaxPool2D(pool_size=(2,2),strides=2,padding='same'),

    Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'),
    Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'),
    MaxPool2D(pool_size=(2,2),strides=2,padding='same'),

    Flatten(),
    Dense(256, activation = 'relu', kernel_regularizer = l2(0.001)),
    Dense(10, activation = 'softmax')
])
model.compile(optimizer = Adam(1e-4), loss = 'sparse_categorical_crossentropy',metrics=['acc'])

history = model.fit(x_train,y_train,epochs=30,batch_size=32,validation_data=(x_val,y_val))
```

    Epoch 1/30
    1094/1094 [==============================] - 21s 9ms/step - loss: 1.8911 - acc: 0.4160 - val_loss: 1.5588 - val_acc: 0.5231
    Epoch 2/30
    1094/1094 [==============================] - 9s 9ms/step - loss: 1.4633 - acc: 0.5561 - val_loss: 1.3681 - val_acc: 0.5793
    Epoch 3/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 1.2645 - acc: 0.6245 - val_loss: 1.2592 - val_acc: 0.6083
    Epoch 4/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 1.1260 - acc: 0.6673 - val_loss: 1.1532 - val_acc: 0.6517
    Epoch 5/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 1.0219 - acc: 0.6986 - val_loss: 1.0304 - val_acc: 0.6903
    Epoch 6/30
    1094/1094 [==============================] - 12s 11ms/step - loss: 0.9312 - acc: 0.7281 - val_loss: 0.9851 - val_acc: 0.7042
    Epoch 7/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.8512 - acc: 0.7549 - val_loss: 0.9919 - val_acc: 0.7039
    Epoch 8/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.7775 - acc: 0.7787 - val_loss: 0.9666 - val_acc: 0.7101
    Epoch 9/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.7101 - acc: 0.7989 - val_loss: 0.9242 - val_acc: 0.7278
    Epoch 10/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.6407 - acc: 0.8236 - val_loss: 0.9867 - val_acc: 0.7102
    Epoch 11/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.5776 - acc: 0.8459 - val_loss: 0.9255 - val_acc: 0.7281
    Epoch 12/30
    1094/1094 [==============================] - 11s 10ms/step - loss: 0.5180 - acc: 0.8642 - val_loss: 0.9973 - val_acc: 0.7243
    Epoch 13/30
    1094/1094 [==============================] - 11s 10ms/step - loss: 0.4594 - acc: 0.8850 - val_loss: 1.0191 - val_acc: 0.7229
    Epoch 14/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.4022 - acc: 0.9031 - val_loss: 1.0690 - val_acc: 0.7244
    Epoch 15/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.3557 - acc: 0.9212 - val_loss: 1.1696 - val_acc: 0.7102
    Epoch 16/30
    1094/1094 [==============================] - 12s 11ms/step - loss: 0.3151 - acc: 0.9332 - val_loss: 1.1834 - val_acc: 0.7153
    Epoch 17/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.2806 - acc: 0.9455 - val_loss: 1.2361 - val_acc: 0.7161
    Epoch 18/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.2482 - acc: 0.9579 - val_loss: 1.2812 - val_acc: 0.7267
    Epoch 19/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.2263 - acc: 0.9638 - val_loss: 1.3865 - val_acc: 0.7163
    Epoch 20/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.2150 - acc: 0.9677 - val_loss: 1.3755 - val_acc: 0.7252
    Epoch 21/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.2061 - acc: 0.9691 - val_loss: 1.4345 - val_acc: 0.7248
    Epoch 22/30
    1094/1094 [==============================] - 8s 8ms/step - loss: 0.1979 - acc: 0.9720 - val_loss: 1.4962 - val_acc: 0.7182
    Epoch 23/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.1859 - acc: 0.9759 - val_loss: 1.5132 - val_acc: 0.7215
    Epoch 24/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.1799 - acc: 0.9776 - val_loss: 1.6924 - val_acc: 0.7135
    Epoch 25/30
    1094/1094 [==============================] - 12s 11ms/step - loss: 0.1744 - acc: 0.9782 - val_loss: 1.6544 - val_acc: 0.7182
    Epoch 26/30
    1094/1094 [==============================] - 9s 9ms/step - loss: 0.1716 - acc: 0.9789 - val_loss: 1.7136 - val_acc: 0.7171
    Epoch 27/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.1679 - acc: 0.9796 - val_loss: 1.6736 - val_acc: 0.7150
    Epoch 28/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.1669 - acc: 0.9797 - val_loss: 1.5413 - val_acc: 0.7316
    Epoch 29/30
    1094/1094 [==============================] - 8s 7ms/step - loss: 0.1563 - acc: 0.9826 - val_loss: 1.7126 - val_acc: 0.7165
    Epoch 30/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.1536 - acc: 0.9830 - val_loss: 1.6584 - val_acc: 0.7255
    


```python
import matplotlib.pyplot as plt

his_dict = history.history
loss = his_dict['loss']
val_loss = his_dict['val_loss']

epochs = range(1, len(loss)+1)
fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(1,2,1)
ax1.plot(epochs,loss,color='blue',label='train_loss')
ax1.plot(epochs,val_loss,color='orange',label='val_loss')
ax1.set_title('loss')
ax1.legend()

acc = his_dict['acc']
val_acc = his_dict['val_acc']

ax2 = fig.add_subplot(1,2,2)
ax2.plot(epochs,acc,color='blue',label='train_acc')
ax2.plot(epochs,val_acc,color='orange',label='val_acc')
ax2.set_title('acc')
ax2.legend()

plt.show()
```


    
![png](10_cnn_regularizer_files/10_cnn_regularizer_2_0.png)
    


* 이전 그래프보다 꺾이는 각도가 더 안정적임
