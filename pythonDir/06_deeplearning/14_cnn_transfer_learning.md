## 전이 학습
* 사전 학습된 네트워크의 가중치를 사용. 크게 세 가지
* 기본 과정 : 입력 -> 모델 -> 분류기 -> 출력
1. 모델을 변형하지 않고 사용
2. 모델 분류기 재학습
3. 모델 일부를 재학습시키기
* 전체 재학습은 시간이 많이 걸리므로 일부를 조정해서 이용한다.


```python
from keras.datasets import cifar10
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
plt.imshow(x_test[0])
x_mean = np.mean(x_train,axis=(0,1,2))
x_std = np.std(x_train, axis = (0,1,2))

x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.3,random_state=777)
y_train.shape
```

    Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    170498071/170498071 [==============================] - 5s 0us/step
    




    (35000, 1)




    
![png](14_cnn_transfer_learning_files/14_cnn_transfer_learning_1_2.png)
    



```python
x_train.shape
```




    (35000, 32, 32, 3)



### 전이 학습 설정하기


```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(horizontal_flip=True, 
                                   zoom_range=0.2, 
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                   rotation_range=30, 
                                   fill_mode='nearest') 

val_datagen = ImageDataGenerator()

batch_size = 32

train_generator = train_datagen.flow(x_train,y_train,batch_size=batch_size)
val_generator = val_datagen.flow(x_val,y_val,batch_size=batch_size)
```


```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation, BatchNormalization
from keras.optimizers import Adam

from keras.applications import VGG16

vgg16 = VGG16(include_top=False,input_shape=(32, 32, 3))
vgg16.summary()
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58889256/58889256 [==============================] - 0s 0us/step
    Model: "vgg16"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 32, 32, 3)]       0         
                                                                     
     block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792      
                                                                     
     block1_conv2 (Conv2D)       (None, 32, 32, 64)        36928     
                                                                     
     block1_pool (MaxPooling2D)  (None, 16, 16, 64)        0         
                                                                     
     block2_conv1 (Conv2D)       (None, 16, 16, 128)       73856     
                                                                     
     block2_conv2 (Conv2D)       (None, 16, 16, 128)       147584    
                                                                     
     block2_pool (MaxPooling2D)  (None, 8, 8, 128)         0         
                                                                     
     block3_conv1 (Conv2D)       (None, 8, 8, 256)         295168    
                                                                     
     block3_conv2 (Conv2D)       (None, 8, 8, 256)         590080    
                                                                     
     block3_conv3 (Conv2D)       (None, 8, 8, 256)         590080    
                                                                     
     block3_pool (MaxPooling2D)  (None, 4, 4, 256)         0         
                                                                     
     block4_conv1 (Conv2D)       (None, 4, 4, 512)         1180160   
                                                                     
     block4_conv2 (Conv2D)       (None, 4, 4, 512)         2359808   
                                                                     
     block4_conv3 (Conv2D)       (None, 4, 4, 512)         2359808   
                                                                     
     block4_pool (MaxPooling2D)  (None, 2, 2, 512)         0         
                                                                     
     block5_conv1 (Conv2D)       (None, 2, 2, 512)         2359808   
                                                                     
     block5_conv2 (Conv2D)       (None, 2, 2, 512)         2359808   
                                                                     
     block5_conv3 (Conv2D)       (None, 2, 2, 512)         2359808   
                                                                     
     block5_pool (MaxPooling2D)  (None, 1, 1, 512)         0         
                                                                     
    =================================================================
    Total params: 14,714,688
    Trainable params: 14,714,688
    Non-trainable params: 0
    _________________________________________________________________
    

### 모델 동결 해제하기


```python
for layer in vgg16.layers[:-4]: # 모델 끝 4개 층만 선택
  layer.trainable = False # 동결을 해제제
```

### 전이 학습을 통해 학습하기


```python
model = Sequential([
    vgg16,
    Flatten(),
    Dense(256),
    BatchNormalization(),
    Activation('relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer = Adam(1e-4), loss = 'sparse_categorical_crossentropy',metrics=['acc'])

def get_step(train_len, batch_size):
  if(train_len % batch_size > 0):
    return train_len // batch_size + 1
  else:
    return train_len // batch_size

history = model.fit(train_generator,
                    epochs=100,
                    steps_per_epoch=get_step(len(x_train),batch_size),
                    validation_data=val_generator,
                    validation_steps = get_step(len(x_val), batch_size))
```

    Epoch 1/100
    1094/1094 [==============================] - 39s 27ms/step - loss: 1.1232 - acc: 0.1066 - val_loss: 0.9254 - val_acc: 0.0801
    Epoch 2/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.9237 - acc: 0.1017 - val_loss: 1.0233 - val_acc: 0.0944
    Epoch 3/100
    1094/1094 [==============================] - 31s 28ms/step - loss: 0.8482 - acc: 0.1021 - val_loss: 0.7274 - val_acc: 0.1223
    Epoch 4/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.7972 - acc: 0.1049 - val_loss: 0.7824 - val_acc: 0.0863
    Epoch 5/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.7713 - acc: 0.1042 - val_loss: 0.7694 - val_acc: 0.0774
    Epoch 6/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.7327 - acc: 0.1036 - val_loss: 0.7249 - val_acc: 0.0872
    Epoch 7/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.7002 - acc: 0.1028 - val_loss: 0.7604 - val_acc: 0.0889
    Epoch 8/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.6862 - acc: 0.1036 - val_loss: 0.6829 - val_acc: 0.0937
    Epoch 9/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.6573 - acc: 0.1030 - val_loss: 0.6741 - val_acc: 0.0921
    Epoch 10/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.6342 - acc: 0.1027 - val_loss: 0.6682 - val_acc: 0.0860
    Epoch 11/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.6124 - acc: 0.1028 - val_loss: 0.7338 - val_acc: 0.0914
    Epoch 12/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.5837 - acc: 0.1029 - val_loss: 0.6624 - val_acc: 0.1075
    Epoch 13/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.5648 - acc: 0.1034 - val_loss: 0.6429 - val_acc: 0.1089
    Epoch 14/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.5460 - acc: 0.1022 - val_loss: 0.7116 - val_acc: 0.0957
    Epoch 15/100
    1094/1094 [==============================] - 30s 27ms/step - loss: 0.5241 - acc: 0.1033 - val_loss: 0.6498 - val_acc: 0.1003
    Epoch 16/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.5094 - acc: 0.1024 - val_loss: 0.6435 - val_acc: 0.0935
    Epoch 17/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.4940 - acc: 0.1030 - val_loss: 0.6541 - val_acc: 0.0895
    Epoch 18/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.4793 - acc: 0.1023 - val_loss: 0.7099 - val_acc: 0.0850
    Epoch 19/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.4558 - acc: 0.1026 - val_loss: 0.6685 - val_acc: 0.1013
    Epoch 20/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.4433 - acc: 0.1026 - val_loss: 0.6734 - val_acc: 0.0885
    Epoch 21/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.4298 - acc: 0.1022 - val_loss: 0.6916 - val_acc: 0.0985
    Epoch 22/100
    1094/1094 [==============================] - 30s 27ms/step - loss: 0.4174 - acc: 0.1027 - val_loss: 0.7218 - val_acc: 0.0895
    Epoch 23/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.4020 - acc: 0.1024 - val_loss: 0.6917 - val_acc: 0.1029
    Epoch 24/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.3852 - acc: 0.1018 - val_loss: 0.6456 - val_acc: 0.0971
    Epoch 25/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.3721 - acc: 0.1017 - val_loss: 0.6781 - val_acc: 0.0975
    Epoch 26/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.3651 - acc: 0.1023 - val_loss: 0.6523 - val_acc: 0.0901
    Epoch 27/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.3525 - acc: 0.1025 - val_loss: 0.7449 - val_acc: 0.0925
    Epoch 28/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.3403 - acc: 0.1025 - val_loss: 0.7459 - val_acc: 0.0991
    Epoch 29/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.3247 - acc: 0.1028 - val_loss: 0.7486 - val_acc: 0.0937
    Epoch 30/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.3230 - acc: 0.1023 - val_loss: 0.7377 - val_acc: 0.0898
    Epoch 31/100
    1094/1094 [==============================] - 28s 26ms/step - loss: 0.3069 - acc: 0.1023 - val_loss: 0.7511 - val_acc: 0.1002
    Epoch 32/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.3065 - acc: 0.1016 - val_loss: 0.7443 - val_acc: 0.0883
    Epoch 33/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.2918 - acc: 0.1021 - val_loss: 0.7615 - val_acc: 0.0917
    Epoch 34/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.2819 - acc: 0.1017 - val_loss: 0.7445 - val_acc: 0.0892
    Epoch 35/100
    1094/1094 [==============================] - 30s 27ms/step - loss: 0.2806 - acc: 0.1017 - val_loss: 0.7865 - val_acc: 0.0869
    Epoch 36/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.2667 - acc: 0.1023 - val_loss: 0.7423 - val_acc: 0.0889
    Epoch 37/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.2609 - acc: 0.1016 - val_loss: 0.7722 - val_acc: 0.0953
    Epoch 38/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.2576 - acc: 0.1013 - val_loss: 0.7579 - val_acc: 0.0941
    Epoch 39/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.2412 - acc: 0.1017 - val_loss: 0.7939 - val_acc: 0.0987
    Epoch 40/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.2430 - acc: 0.1025 - val_loss: 0.8261 - val_acc: 0.0903
    Epoch 41/100
    1094/1094 [==============================] - 30s 27ms/step - loss: 0.2356 - acc: 0.1019 - val_loss: 0.8564 - val_acc: 0.0867
    Epoch 42/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.2379 - acc: 0.1016 - val_loss: 0.7903 - val_acc: 0.0989
    Epoch 43/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.2251 - acc: 0.1025 - val_loss: 0.7659 - val_acc: 0.0863
    Epoch 44/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.2222 - acc: 0.1021 - val_loss: 0.8367 - val_acc: 0.0897
    Epoch 45/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.2164 - acc: 0.1017 - val_loss: 0.8442 - val_acc: 0.0958
    Epoch 46/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.2081 - acc: 0.1019 - val_loss: 0.8297 - val_acc: 0.0917
    Epoch 47/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.2046 - acc: 0.1015 - val_loss: 0.8542 - val_acc: 0.0928
    Epoch 48/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.2079 - acc: 0.1019 - val_loss: 0.8617 - val_acc: 0.0951
    Epoch 49/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.2000 - acc: 0.1020 - val_loss: 0.8783 - val_acc: 0.0878
    Epoch 50/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1983 - acc: 0.1020 - val_loss: 0.8454 - val_acc: 0.0953
    Epoch 51/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1873 - acc: 0.1023 - val_loss: 0.8572 - val_acc: 0.0943
    Epoch 52/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1922 - acc: 0.1019 - val_loss: 0.8840 - val_acc: 0.0925
    Epoch 53/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1828 - acc: 0.1024 - val_loss: 0.8536 - val_acc: 0.0771
    Epoch 54/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.1796 - acc: 0.1023 - val_loss: 0.8774 - val_acc: 0.0925
    Epoch 55/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1767 - acc: 0.1019 - val_loss: 0.8597 - val_acc: 0.0965
    Epoch 56/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1754 - acc: 0.1018 - val_loss: 0.8955 - val_acc: 0.0958
    Epoch 57/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1687 - acc: 0.1023 - val_loss: 0.8777 - val_acc: 0.0927
    Epoch 58/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1697 - acc: 0.1019 - val_loss: 0.9341 - val_acc: 0.0861
    Epoch 59/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.1644 - acc: 0.1020 - val_loss: 0.9230 - val_acc: 0.0856
    Epoch 60/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.1621 - acc: 0.1016 - val_loss: 0.8926 - val_acc: 0.0951
    Epoch 61/100
    1094/1094 [==============================] - 30s 28ms/step - loss: 0.1625 - acc: 0.1020 - val_loss: 0.8954 - val_acc: 0.0915
    Epoch 62/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1547 - acc: 0.1020 - val_loss: 0.8672 - val_acc: 0.0881
    Epoch 63/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1532 - acc: 0.1019 - val_loss: 0.9040 - val_acc: 0.0961
    Epoch 64/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1492 - acc: 0.1015 - val_loss: 0.9294 - val_acc: 0.1021
    Epoch 65/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1510 - acc: 0.1018 - val_loss: 0.8497 - val_acc: 0.0940
    Epoch 66/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.1498 - acc: 0.1021 - val_loss: 0.8973 - val_acc: 0.0865
    Epoch 67/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.1471 - acc: 0.1017 - val_loss: 0.9375 - val_acc: 0.0899
    Epoch 68/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1412 - acc: 0.1019 - val_loss: 0.8779 - val_acc: 0.1035
    Epoch 69/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1328 - acc: 0.1020 - val_loss: 0.9605 - val_acc: 0.0898
    Epoch 70/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1438 - acc: 0.1015 - val_loss: 0.8817 - val_acc: 0.0945
    Epoch 71/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1389 - acc: 0.1020 - val_loss: 0.9232 - val_acc: 0.0886
    Epoch 72/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1321 - acc: 0.1018 - val_loss: 0.9600 - val_acc: 0.0970
    Epoch 73/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1307 - acc: 0.1019 - val_loss: 0.9581 - val_acc: 0.0926
    Epoch 74/100
    1094/1094 [==============================] - 30s 27ms/step - loss: 0.1255 - acc: 0.1018 - val_loss: 0.9524 - val_acc: 0.0966
    Epoch 75/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1296 - acc: 0.1013 - val_loss: 0.9040 - val_acc: 0.1030
    Epoch 76/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1309 - acc: 0.1023 - val_loss: 0.9377 - val_acc: 0.0887
    Epoch 77/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1221 - acc: 0.1018 - val_loss: 1.1087 - val_acc: 0.1040
    Epoch 78/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1298 - acc: 0.1023 - val_loss: 0.9283 - val_acc: 0.0972
    Epoch 79/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1245 - acc: 0.1021 - val_loss: 0.9741 - val_acc: 0.0874
    Epoch 80/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1199 - acc: 0.1018 - val_loss: 0.9564 - val_acc: 0.0979
    Epoch 81/100
    1094/1094 [==============================] - 30s 27ms/step - loss: 0.1197 - acc: 0.1019 - val_loss: 0.9692 - val_acc: 0.1055
    Epoch 82/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1168 - acc: 0.1022 - val_loss: 0.9408 - val_acc: 0.0879
    Epoch 83/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1165 - acc: 0.1022 - val_loss: 0.9622 - val_acc: 0.0863
    Epoch 84/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1158 - acc: 0.1018 - val_loss: 0.9181 - val_acc: 0.0922
    Epoch 85/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1139 - acc: 0.1020 - val_loss: 0.9435 - val_acc: 0.0904
    Epoch 86/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1145 - acc: 0.1022 - val_loss: 1.0401 - val_acc: 0.0867
    Epoch 87/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1171 - acc: 0.1025 - val_loss: 0.9640 - val_acc: 0.0889
    Epoch 88/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1076 - acc: 0.1018 - val_loss: 1.0060 - val_acc: 0.0927
    Epoch 89/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1074 - acc: 0.1020 - val_loss: 1.0566 - val_acc: 0.0877
    Epoch 90/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1108 - acc: 0.1017 - val_loss: 1.0224 - val_acc: 0.0865
    Epoch 91/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1042 - acc: 0.1021 - val_loss: 0.9907 - val_acc: 0.0999
    Epoch 92/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1064 - acc: 0.1025 - val_loss: 0.9853 - val_acc: 0.0883
    Epoch 93/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1037 - acc: 0.1021 - val_loss: 0.9927 - val_acc: 0.0935
    Epoch 94/100
    1094/1094 [==============================] - 30s 27ms/step - loss: 0.1027 - acc: 0.1022 - val_loss: 1.0342 - val_acc: 0.0918
    Epoch 95/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1033 - acc: 0.1019 - val_loss: 0.9990 - val_acc: 0.0903
    Epoch 96/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1062 - acc: 0.1017 - val_loss: 0.9936 - val_acc: 0.0868
    Epoch 97/100
    1094/1094 [==============================] - 29s 27ms/step - loss: 0.0961 - acc: 0.1021 - val_loss: 0.9708 - val_acc: 0.1011
    Epoch 98/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1023 - acc: 0.1021 - val_loss: 0.9753 - val_acc: 0.0968
    Epoch 99/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.1001 - acc: 0.1019 - val_loss: 0.9992 - val_acc: 0.1010
    Epoch 100/100
    1094/1094 [==============================] - 30s 27ms/step - loss: 0.0990 - acc: 0.1019 - val_loss: 1.0171 - val_acc: 0.0897
    
