### 드롭아웃
* 과대적합을 피하기 위해 학습동안 일부 유닛을 제외(드롭)함.


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
    170498071/170498071 [==============================] - 15s 0us/step
    


```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

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

    Dropout(0.2), 
    Flatten(),
    Dense(256, activation = 'relu'),
    Dense(10, activation = 'softmax')
])
model.compile(optimizer = Adam(1e-4), loss = 'sparse_categorical_crossentropy',metrics=['acc'])

history = model.fit(x_train,y_train,epochs=30,batch_size=32,validation_data=(x_val,y_val))
```

    Epoch 1/30
    1094/1094 [==============================] - 17s 7ms/step - loss: 1.6590 - acc: 0.3996 - val_loss: 1.3811 - val_acc: 0.4973
    Epoch 2/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 1.3029 - acc: 0.5350 - val_loss: 1.1778 - val_acc: 0.5735
    Epoch 3/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 1.1445 - acc: 0.5943 - val_loss: 1.0843 - val_acc: 0.6103
    Epoch 4/30
    1094/1094 [==============================] - 7s 7ms/step - loss: 1.0230 - acc: 0.6399 - val_loss: 0.9949 - val_acc: 0.6442
    Epoch 5/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.9309 - acc: 0.6732 - val_loss: 0.9176 - val_acc: 0.6755
    Epoch 6/30
    1094/1094 [==============================] - 8s 7ms/step - loss: 0.8512 - acc: 0.6989 - val_loss: 0.8756 - val_acc: 0.6886
    Epoch 7/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.7809 - acc: 0.7272 - val_loss: 0.8630 - val_acc: 0.6974
    Epoch 8/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.7159 - acc: 0.7495 - val_loss: 0.8081 - val_acc: 0.7158
    Epoch 9/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.6599 - acc: 0.7690 - val_loss: 0.7854 - val_acc: 0.7227
    Epoch 10/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.6110 - acc: 0.7880 - val_loss: 0.7766 - val_acc: 0.7350
    Epoch 11/30
    1094/1094 [==============================] - 7s 7ms/step - loss: 0.5558 - acc: 0.8059 - val_loss: 0.7665 - val_acc: 0.7364
    Epoch 12/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.5124 - acc: 0.8214 - val_loss: 0.7620 - val_acc: 0.7415
    Epoch 13/30
    1094/1094 [==============================] - 7s 7ms/step - loss: 0.4705 - acc: 0.8347 - val_loss: 0.7523 - val_acc: 0.7468
    Epoch 14/30
    1094/1094 [==============================] - 7s 7ms/step - loss: 0.4296 - acc: 0.8488 - val_loss: 0.7621 - val_acc: 0.7439
    Epoch 15/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.3873 - acc: 0.8635 - val_loss: 0.8044 - val_acc: 0.7427
    Epoch 16/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.3545 - acc: 0.8759 - val_loss: 0.8052 - val_acc: 0.7459
    Epoch 17/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.3194 - acc: 0.8883 - val_loss: 0.8196 - val_acc: 0.7457
    Epoch 18/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.2874 - acc: 0.8996 - val_loss: 0.8225 - val_acc: 0.7465
    Epoch 19/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.2572 - acc: 0.9099 - val_loss: 0.8374 - val_acc: 0.7505
    Epoch 20/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.2296 - acc: 0.9203 - val_loss: 0.8724 - val_acc: 0.7492
    Epoch 21/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.2121 - acc: 0.9252 - val_loss: 0.9286 - val_acc: 0.7443
    Epoch 22/30
    1094/1094 [==============================] - 8s 8ms/step - loss: 0.1931 - acc: 0.9331 - val_loss: 0.9525 - val_acc: 0.7451
    Epoch 23/30
    1094/1094 [==============================] - 7s 7ms/step - loss: 0.1658 - acc: 0.9420 - val_loss: 1.0350 - val_acc: 0.7436
    Epoch 24/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.1587 - acc: 0.9447 - val_loss: 1.0406 - val_acc: 0.7421
    Epoch 25/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.1381 - acc: 0.9510 - val_loss: 0.9914 - val_acc: 0.7475
    Epoch 26/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.1302 - acc: 0.9542 - val_loss: 1.0216 - val_acc: 0.7521
    Epoch 27/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.1214 - acc: 0.9573 - val_loss: 1.0605 - val_acc: 0.7455
    Epoch 28/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.1084 - acc: 0.9623 - val_loss: 1.0363 - val_acc: 0.7561
    Epoch 29/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.1062 - acc: 0.9633 - val_loss: 1.0809 - val_acc: 0.7517
    Epoch 30/30
    1094/1094 [==============================] - 7s 6ms/step - loss: 0.0933 - acc: 0.9675 - val_loss: 1.1280 - val_acc: 0.7526
    


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


    
![png](11_cnn_dropout_files/11_cnn_dropout_3_0.png)
    



```python

```
