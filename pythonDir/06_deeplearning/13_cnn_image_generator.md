### 데이터 증식 사용하기
* 기존 데이터를 변형하여 새로운 데이터를 만들어내는 방법
* 일반화를 보완하지만 근본적인 해결책은 아님
* 데이터가 다양해지므로 테스트 성능이 높아짐.
* 케라스가 이미지 제너레이터(ImageGenerator)를 제공.


```python
from keras.utils import load_img, img_to_array # 케라스가 업데이트되면서 경로가 바뀌었다.
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
```


```python
train_datagen = ImageDataGenerator(horizontal_flip=True, # 이미지 수평 방향 뒤집기
                                   vertical_flip=True, # 이미지 수직 방향 뒤집기
                                   shear_range=0.5, # 밀림 강도를 50%조절
                                   brightness_range=[0.5,1.5], # 밝기를 0.5~1.5로 조절
                                   zoom_range=0.2, # 확대 비율 20%
                                   width_shift_range=0.1, # 너비 방향 이동 10%
                                   height_shift_range=0.1, # 높이 방향 이동 10%
                                   rotation_range=30, # 이미지 회전 30도
                                   fill_mode='nearest') # 이미지 변환시 픽셀 변환을 근처를 가져옴
```


```python
img = img_to_array(load_img('img04.jpg')).astype(np.uint8)
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x7f3ca01915b0>




    
![png](13_cnn_image_generator_files/13_cnn_image_generator_3_1.png)
    



```python
result = img.reshape((1,)+img.shape)
img.shape,result.shape
```




    ((460, 728, 3), (1, 460, 728, 3))




```python
train_generator = train_datagen.flow(result,batch_size=1)
```


```python
fig = plt.figure(figsize=(5,5))
fig.suptitle('증강 이미지')

for i in range(9):
  data = next(train_generator)
  image = data[0]
  plt.subplot(3,3,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(np.array(image,dtype=np.uint8))
plt.show()
```

    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:214: RuntimeWarning: Glyph 51613 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:214: RuntimeWarning: Glyph 44053 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:214: RuntimeWarning: Glyph 51060 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:214: RuntimeWarning: Glyph 48120 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:214: RuntimeWarning: Glyph 51648 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:183: RuntimeWarning: Glyph 51613 missing from current font.
      font.set_text(s, 0, flags=flags)
    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:183: RuntimeWarning: Glyph 44053 missing from current font.
      font.set_text(s, 0, flags=flags)
    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:183: RuntimeWarning: Glyph 51060 missing from current font.
      font.set_text(s, 0, flags=flags)
    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:183: RuntimeWarning: Glyph 48120 missing from current font.
      font.set_text(s, 0, flags=flags)
    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:183: RuntimeWarning: Glyph 51648 missing from current font.
      font.set_text(s, 0, flags=flags)
    


    
![png](13_cnn_image_generator_files/13_cnn_image_generator_6_1.png)
    



```python
from keras.datasets import cifar10
import numpy as np
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_mean = np.mean(x_train,axis=(0,1,2))
x_std = np.std(x_train, axis = (0,1,2))

x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.3,random_state=777)
```


```python
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
    1094/1094 [==============================] - 24s 21ms/step - loss: 1.6084 - acc: 0.1101 - val_loss: 1.3281 - val_acc: 0.0959
    Epoch 2/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 1.3403 - acc: 0.1031 - val_loss: 1.3337 - val_acc: 0.1547
    Epoch 3/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 1.2327 - acc: 0.1030 - val_loss: 1.1966 - val_acc: 0.1062
    Epoch 4/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 1.1464 - acc: 0.1057 - val_loss: 1.1078 - val_acc: 0.0997
    Epoch 5/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 1.0812 - acc: 0.1055 - val_loss: 1.0105 - val_acc: 0.1137
    Epoch 6/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 1.0271 - acc: 0.1056 - val_loss: 0.9308 - val_acc: 0.1032
    Epoch 7/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.9786 - acc: 0.1053 - val_loss: 1.0202 - val_acc: 0.0892
    Epoch 8/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.9338 - acc: 0.1044 - val_loss: 0.8741 - val_acc: 0.0925
    Epoch 9/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.9059 - acc: 0.1044 - val_loss: 0.9413 - val_acc: 0.0959
    Epoch 10/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.8722 - acc: 0.1038 - val_loss: 0.9131 - val_acc: 0.0842
    Epoch 11/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.8507 - acc: 0.1030 - val_loss: 0.7473 - val_acc: 0.1023
    Epoch 12/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.8207 - acc: 0.1045 - val_loss: 0.7977 - val_acc: 0.0937
    Epoch 13/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.8058 - acc: 0.1031 - val_loss: 0.7455 - val_acc: 0.0823
    Epoch 14/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.7749 - acc: 0.1038 - val_loss: 0.7972 - val_acc: 0.0890
    Epoch 15/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.7565 - acc: 0.1041 - val_loss: 0.7312 - val_acc: 0.1167
    Epoch 16/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.7383 - acc: 0.1040 - val_loss: 0.8686 - val_acc: 0.0857
    Epoch 17/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.7287 - acc: 0.1051 - val_loss: 0.6603 - val_acc: 0.1153
    Epoch 18/100
    1094/1094 [==============================] - 24s 21ms/step - loss: 0.7118 - acc: 0.1048 - val_loss: 0.6826 - val_acc: 0.0921
    Epoch 19/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.6930 - acc: 0.1037 - val_loss: 0.6209 - val_acc: 0.1015
    Epoch 20/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.6759 - acc: 0.1039 - val_loss: 0.6097 - val_acc: 0.0922
    Epoch 21/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.6650 - acc: 0.1039 - val_loss: 0.6864 - val_acc: 0.0851
    Epoch 22/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.6515 - acc: 0.1047 - val_loss: 0.7263 - val_acc: 0.0904
    Epoch 23/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.6357 - acc: 0.1046 - val_loss: 0.5968 - val_acc: 0.0885
    Epoch 24/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.6271 - acc: 0.1031 - val_loss: 0.6101 - val_acc: 0.1044
    Epoch 25/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.6218 - acc: 0.1046 - val_loss: 0.6375 - val_acc: 0.0991
    Epoch 26/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.6110 - acc: 0.1034 - val_loss: 0.6061 - val_acc: 0.1021
    Epoch 27/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.5961 - acc: 0.1039 - val_loss: 0.5982 - val_acc: 0.0889
    Epoch 28/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.5949 - acc: 0.1041 - val_loss: 0.6364 - val_acc: 0.0925
    Epoch 29/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.5801 - acc: 0.1039 - val_loss: 0.5903 - val_acc: 0.0930
    Epoch 30/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.5755 - acc: 0.1042 - val_loss: 0.6600 - val_acc: 0.1137
    Epoch 31/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.5586 - acc: 0.1032 - val_loss: 0.5758 - val_acc: 0.0809
    Epoch 32/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.5535 - acc: 0.1039 - val_loss: 0.5542 - val_acc: 0.1069
    Epoch 33/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.5429 - acc: 0.1033 - val_loss: 0.5902 - val_acc: 0.1066
    Epoch 34/100
    1094/1094 [==============================] - 24s 21ms/step - loss: 0.5404 - acc: 0.1040 - val_loss: 0.5369 - val_acc: 0.0805
    Epoch 35/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.5356 - acc: 0.1034 - val_loss: 0.5613 - val_acc: 0.0892
    Epoch 36/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.5234 - acc: 0.1041 - val_loss: 0.5346 - val_acc: 0.0883
    Epoch 37/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.5197 - acc: 0.1030 - val_loss: 0.6350 - val_acc: 0.0845
    Epoch 38/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.5052 - acc: 0.1042 - val_loss: 0.5946 - val_acc: 0.1013
    Epoch 39/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.5074 - acc: 0.1041 - val_loss: 0.5864 - val_acc: 0.0903
    Epoch 40/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.5035 - acc: 0.1042 - val_loss: 0.5870 - val_acc: 0.0997
    Epoch 41/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.4955 - acc: 0.1045 - val_loss: 0.5151 - val_acc: 0.0914
    Epoch 42/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.4908 - acc: 0.1039 - val_loss: 0.5293 - val_acc: 0.0887
    Epoch 43/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.4817 - acc: 0.1033 - val_loss: 0.5646 - val_acc: 0.0975
    Epoch 44/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.4804 - acc: 0.1039 - val_loss: 0.5483 - val_acc: 0.1133
    Epoch 45/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.4777 - acc: 0.1038 - val_loss: 0.5321 - val_acc: 0.1070
    Epoch 46/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.4673 - acc: 0.1031 - val_loss: 0.5112 - val_acc: 0.0883
    Epoch 47/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.4557 - acc: 0.1024 - val_loss: 0.4835 - val_acc: 0.1021
    Epoch 48/100
    1094/1094 [==============================] - 24s 21ms/step - loss: 0.4562 - acc: 0.1041 - val_loss: 0.5352 - val_acc: 0.0911
    Epoch 49/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.4491 - acc: 0.1031 - val_loss: 0.4676 - val_acc: 0.0990
    Epoch 50/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.4483 - acc: 0.1033 - val_loss: 0.5136 - val_acc: 0.0970
    Epoch 51/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.4388 - acc: 0.1035 - val_loss: 0.5382 - val_acc: 0.0848
    Epoch 52/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.4390 - acc: 0.1029 - val_loss: 0.5263 - val_acc: 0.1049
    Epoch 53/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.4380 - acc: 0.1034 - val_loss: 0.5370 - val_acc: 0.0903
    Epoch 54/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.4284 - acc: 0.1025 - val_loss: 0.4899 - val_acc: 0.1068
    Epoch 55/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.4274 - acc: 0.1030 - val_loss: 0.5376 - val_acc: 0.0943
    Epoch 56/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.4208 - acc: 0.1034 - val_loss: 0.5024 - val_acc: 0.1047
    Epoch 57/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.4189 - acc: 0.1050 - val_loss: 0.4721 - val_acc: 0.0902
    Epoch 58/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.4139 - acc: 0.1041 - val_loss: 0.5178 - val_acc: 0.1069
    Epoch 59/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.4100 - acc: 0.1029 - val_loss: 0.5212 - val_acc: 0.0827
    Epoch 60/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.4013 - acc: 0.1039 - val_loss: 0.4946 - val_acc: 0.0889
    Epoch 61/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.4088 - acc: 0.1029 - val_loss: 0.4953 - val_acc: 0.1046
    Epoch 62/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.4060 - acc: 0.1032 - val_loss: 0.4357 - val_acc: 0.1017
    Epoch 63/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3976 - acc: 0.1031 - val_loss: 0.5213 - val_acc: 0.0959
    Epoch 64/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3909 - acc: 0.1034 - val_loss: 0.5175 - val_acc: 0.1114
    Epoch 65/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3879 - acc: 0.1025 - val_loss: 0.4403 - val_acc: 0.0973
    Epoch 66/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3835 - acc: 0.1035 - val_loss: 0.5952 - val_acc: 0.1074
    Epoch 67/100
    1094/1094 [==============================] - 25s 22ms/step - loss: 0.3813 - acc: 0.1032 - val_loss: 0.5052 - val_acc: 0.0943
    Epoch 68/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.3777 - acc: 0.1035 - val_loss: 0.4995 - val_acc: 0.0880
    Epoch 69/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.3753 - acc: 0.1030 - val_loss: 0.4736 - val_acc: 0.0924
    Epoch 70/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3718 - acc: 0.1029 - val_loss: 0.5046 - val_acc: 0.1020
    Epoch 71/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.3687 - acc: 0.1022 - val_loss: 0.5152 - val_acc: 0.0996
    Epoch 72/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.3703 - acc: 0.1037 - val_loss: 0.5042 - val_acc: 0.0929
    Epoch 73/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3600 - acc: 0.1025 - val_loss: 0.4403 - val_acc: 0.0893
    Epoch 74/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3588 - acc: 0.1039 - val_loss: 0.4919 - val_acc: 0.1005
    Epoch 75/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3589 - acc: 0.1030 - val_loss: 0.4703 - val_acc: 0.0985
    Epoch 76/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3564 - acc: 0.1030 - val_loss: 0.4695 - val_acc: 0.0834
    Epoch 77/100
    1094/1094 [==============================] - 24s 21ms/step - loss: 0.3593 - acc: 0.1027 - val_loss: 0.4600 - val_acc: 0.1109
    Epoch 78/100
    1094/1094 [==============================] - 24s 21ms/step - loss: 0.3522 - acc: 0.1037 - val_loss: 0.5029 - val_acc: 0.1028
    Epoch 79/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.3464 - acc: 0.1035 - val_loss: 0.4629 - val_acc: 0.0863
    Epoch 80/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3398 - acc: 0.1031 - val_loss: 0.4758 - val_acc: 0.0921
    Epoch 81/100
    1094/1094 [==============================] - 23s 21ms/step - loss: 0.3448 - acc: 0.1028 - val_loss: 0.4916 - val_acc: 0.0972
    Epoch 82/100
    1094/1094 [==============================] - 25s 22ms/step - loss: 0.3398 - acc: 0.1032 - val_loss: 0.5095 - val_acc: 0.0925
    Epoch 83/100
    1094/1094 [==============================] - 25s 22ms/step - loss: 0.3354 - acc: 0.1032 - val_loss: 0.4628 - val_acc: 0.0983
    Epoch 84/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3281 - acc: 0.1029 - val_loss: 0.5071 - val_acc: 0.0955
    Epoch 85/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3337 - acc: 0.1032 - val_loss: 0.5060 - val_acc: 0.0974
    Epoch 86/100
    1094/1094 [==============================] - 30s 27ms/step - loss: 0.3280 - acc: 0.1026 - val_loss: 0.4768 - val_acc: 0.1054
    Epoch 87/100
    1094/1094 [==============================] - 38s 34ms/step - loss: 0.3250 - acc: 0.1029 - val_loss: 0.4467 - val_acc: 0.0919
    Epoch 88/100
    1094/1094 [==============================] - 34s 31ms/step - loss: 0.3264 - acc: 0.1026 - val_loss: 0.4769 - val_acc: 0.0940
    Epoch 89/100
    1094/1094 [==============================] - 29s 26ms/step - loss: 0.3198 - acc: 0.1026 - val_loss: 0.4760 - val_acc: 0.1011
    Epoch 90/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3236 - acc: 0.1032 - val_loss: 0.5104 - val_acc: 0.1017
    Epoch 91/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3148 - acc: 0.1027 - val_loss: 0.5031 - val_acc: 0.0934
    Epoch 92/100
    1094/1094 [==============================] - 25s 22ms/step - loss: 0.3166 - acc: 0.1022 - val_loss: 0.5186 - val_acc: 0.0982
    Epoch 93/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3143 - acc: 0.1030 - val_loss: 0.4589 - val_acc: 0.0871
    Epoch 94/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3127 - acc: 0.1035 - val_loss: 0.4232 - val_acc: 0.0873
    Epoch 95/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3066 - acc: 0.1021 - val_loss: 0.4564 - val_acc: 0.0927
    Epoch 96/100
    1094/1094 [==============================] - 25s 23ms/step - loss: 0.3076 - acc: 0.1035 - val_loss: 0.5093 - val_acc: 0.0798
    Epoch 97/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3025 - acc: 0.1032 - val_loss: 0.4478 - val_acc: 0.0961
    Epoch 98/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3000 - acc: 0.1032 - val_loss: 0.4593 - val_acc: 0.0977
    Epoch 99/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.2939 - acc: 0.1025 - val_loss: 0.4793 - val_acc: 0.1139
    Epoch 100/100
    1094/1094 [==============================] - 24s 22ms/step - loss: 0.3033 - acc: 0.1030 - val_loss: 0.4288 - val_acc: 0.0973
    


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


    
![png](13_cnn_image_generator_files/13_cnn_image_generator_10_0.png)
    



```python
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow(x_test,batch_size=batch_size)
pred = model.predict(x_test)
np.argmax(np.round(pred[0],2))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-0f8789a5b24c> in <module>
    ----> 1 test_datagen = ImageDataGenerator()
          2 test_generator = test_datagen.flow(x_test,batch_size=batch_size)
          3 pred = model.predict(x_test)
          4 np.argmax(np.round(pred[0],2))
    

    NameError: name 'ImageDataGenerator' is not defined



```python

```
