### 배치 정규화
* 직접적으로 과대적합을 피하기 위한 방법은 아니지만, 드롭아웃과 비교가 많이 됨


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


```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation, BatchNormalization
from keras.optimizers import Adam

model = Sequential([
    Conv2D(filters=32,kernel_size=3,padding='same',input_shape=(32, 32, 3)),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(filters=32,kernel_size=3,padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2,2),strides=2,padding='same'),

    Conv2D(filters=64,kernel_size=3,padding='same'),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(filters=64,kernel_size=3,padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2,2),strides=2,padding='same'),

    Conv2D(filters=128,kernel_size=3,padding='same'),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(filters=128,kernel_size=3,padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2,2),strides=2,padding='same'),

    Flatten(),
    Dense(256),
    BatchNormalization(),
    Activation('relu'),
    Dense(10, activation = 'softmax')
])
model.compile(optimizer = Adam(1e-4), loss = 'sparse_categorical_crossentropy',metrics=['acc'])

history = model.fit(x_train,y_train,epochs=30,batch_size=32,validation_data=(x_val,y_val))
```

    Epoch 1/30
    1094/1094 [==============================] - 16s 10ms/step - loss: 1.4305 - acc: 0.4954 - val_loss: 1.2480 - val_acc: 0.5548
    Epoch 2/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.9997 - acc: 0.6507 - val_loss: 1.0756 - val_acc: 0.6189
    Epoch 3/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.8003 - acc: 0.7263 - val_loss: 0.9694 - val_acc: 0.6571
    Epoch 4/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.6457 - acc: 0.7853 - val_loss: 0.9279 - val_acc: 0.6759
    Epoch 5/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.5114 - acc: 0.8365 - val_loss: 0.9082 - val_acc: 0.6867
    Epoch 6/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.3927 - acc: 0.8791 - val_loss: 1.0499 - val_acc: 0.6601
    Epoch 7/30
    1094/1094 [==============================] - 9s 9ms/step - loss: 0.2968 - acc: 0.9140 - val_loss: 0.9894 - val_acc: 0.6790
    Epoch 8/30
    1094/1094 [==============================] - 11s 10ms/step - loss: 0.2194 - acc: 0.9424 - val_loss: 1.0835 - val_acc: 0.6664
    Epoch 9/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.1681 - acc: 0.9566 - val_loss: 1.0983 - val_acc: 0.6771
    Epoch 10/30
    1094/1094 [==============================] - 11s 10ms/step - loss: 0.1254 - acc: 0.9707 - val_loss: 1.1121 - val_acc: 0.6729
    Epoch 11/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.1097 - acc: 0.9726 - val_loss: 1.2213 - val_acc: 0.6662
    Epoch 12/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.0903 - acc: 0.9781 - val_loss: 1.2135 - val_acc: 0.6769
    Epoch 13/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.0828 - acc: 0.9795 - val_loss: 1.2837 - val_acc: 0.6713
    Epoch 14/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.0720 - acc: 0.9816 - val_loss: 1.3054 - val_acc: 0.6658
    Epoch 15/30
    1094/1094 [==============================] - 11s 10ms/step - loss: 0.0647 - acc: 0.9831 - val_loss: 1.3889 - val_acc: 0.6612
    Epoch 16/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.0659 - acc: 0.9809 - val_loss: 1.4789 - val_acc: 0.6543
    Epoch 17/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.0563 - acc: 0.9846 - val_loss: 1.3451 - val_acc: 0.6767
    Epoch 18/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.0552 - acc: 0.9851 - val_loss: 1.5027 - val_acc: 0.6615
    Epoch 19/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.0538 - acc: 0.9845 - val_loss: 1.4139 - val_acc: 0.6742
    Epoch 20/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.0477 - acc: 0.9865 - val_loss: 1.4185 - val_acc: 0.6746
    Epoch 21/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.0485 - acc: 0.9857 - val_loss: 1.5418 - val_acc: 0.6631
    Epoch 22/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.0535 - acc: 0.9841 - val_loss: 1.5157 - val_acc: 0.6726
    Epoch 23/30
    1094/1094 [==============================] - 9s 9ms/step - loss: 0.0432 - acc: 0.9876 - val_loss: 1.4211 - val_acc: 0.6829
    Epoch 24/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.0409 - acc: 0.9879 - val_loss: 1.5588 - val_acc: 0.6611
    Epoch 25/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.0446 - acc: 0.9867 - val_loss: 1.4927 - val_acc: 0.6811
    Epoch 26/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.0455 - acc: 0.9859 - val_loss: 1.4337 - val_acc: 0.6836
    Epoch 27/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.0358 - acc: 0.9894 - val_loss: 1.5163 - val_acc: 0.6777
    Epoch 28/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.0361 - acc: 0.9897 - val_loss: 1.5000 - val_acc: 0.6819
    Epoch 29/30
    1094/1094 [==============================] - 9s 8ms/step - loss: 0.0411 - acc: 0.9868 - val_loss: 1.5814 - val_acc: 0.6723
    Epoch 30/30
    1094/1094 [==============================] - 10s 9ms/step - loss: 0.0402 - acc: 0.9875 - val_loss: 1.5313 - val_acc: 0.6766
    


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


    
![png](12_cnn_bn_files/12_cnn_bn_3_0.png)
    



```python

```
