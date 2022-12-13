## Fashion-mnist 살펴보기
* 각 레이블에 해당하는 의류 품목 살펴보기


```python
from keras.datasets.fashion_mnist import load_data
```


```python
# fashion-mnist 다운받기
(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape,x_test.shape)
```

    (60000, 28, 28) (10000, 28, 28)
    


```python
# fashion-mnist 항목 확인하기
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(777)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 이제 확률값으로 나오는 것을 확인함
# 위치 값을 찾아서 구별해냄
sample_size = 9
random_idx = np.random.randint(60000, size=sample_size)

plt.figure(figsize = (5, 5))
for i, idx in enumerate(random_idx):
  plt.subplot(3, 3, i+1) # 3행 3열에 i값을 넣는데 0부터 시작하므로 1을 더해줌
  plt.xticks([]) # x축 눈금관련 정보. 빈 리스트를 넣어서 지워준다.
  plt.yticks([]) # y축
  plt.imshow(x_train[idx], cmap='gray') # index를 학습, cmap을 grayscale로 해줌
  plt.xlabel(class_names[y_train[idx]]) # 레이블을 가져옴
plt.show()
```


    
![png](04_fashion_mnist_files/04_fashion_mnist_3_0.png)
    



```python
x_train.min(), x_train.max()
```




    (0, 255)




```python
x_train = x_train/255
x_test = x_test/255
```


```python
x_train.min(), x_train.max()
```




    (0.0, 1.0)




```python
y_train.min(),y_train.max()
```




    (0, 9)




```python
from keras.utils import to_categorical
```


```python
# 데이터 레이블을 범주형 형태로 변경
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```


```python
y_train.min(),y_train.max()
```




    (0.0, 1.0)




```python
y_train[0]
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.], dtype=float32)




```python
# 검증용 데이터셋 만들기
from sklearn.model_selection import train_test_split
```


```python
# 학습/데이터 비율은 7:3으로 설정
x_train, x_val,y_train,y_val = train_test_split(x_train,y_train,
                                                        test_size=0.3,random_state=777)
```


```python
from keras.models import Sequential
from keras.layers import Dense, Flatten
```

* Flatten : 배치 크기를 제외하고 데이터를 1차원 배열의 형태로 변환해줌
   ex) (128, 6, 2, 2) 입력 -> (128, 24)


```python
first_model = Sequential()
# 입력 데이터의 형태 명시
first_model.add(Flatten(input_shape=(28,28))) # (28, 28) -> (28 * 28)
first_model.add(Dense(64, activation = 'relu')) # 64개의 출력을 가지는 Dense층
first_model.add(Dense(32, activation = 'relu')) # 32개의 출력을 가지는 Dense층
first_model.add(Dense(10, activation = 'softmax')) # 10개의 출력을 가지는 신경망
```


```python
first_model.compile(optimizer='adam', # 옵티마이저 : adam
                          loss='categorical_crossentropy', # 손실함수 : categorical_crossentropy
                          metrics=['acc']) # 평가지표 : acc
```


```python
first_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten (Flatten)           (None, 784)               0         
                                                                     
     dense (Dense)               (None, 64)                50240     
                                                                     
     dense_1 (Dense)             (None, 32)                2080      
                                                                     
     dense_2 (Dense)             (None, 10)                330       
                                                                     
    =================================================================
    Total params: 52,650
    Trainable params: 52,650
    Non-trainable params: 0
    _________________________________________________________________
    


```python
first_history = first_model.fit(x_train, y_train, 
                                    epochs=30, batch_size=128, 
                                    validation_data=(x_val, y_val))
```

    Epoch 1/30
    329/329 [==============================] - 5s 12ms/step - loss: 0.6675 - acc: 0.7718 - val_loss: 0.4710 - val_acc: 0.8411
    Epoch 2/30
    329/329 [==============================] - 3s 10ms/step - loss: 0.4476 - acc: 0.8412 - val_loss: 0.4334 - val_acc: 0.8452
    Epoch 3/30
    329/329 [==============================] - 4s 11ms/step - loss: 0.4094 - acc: 0.8545 - val_loss: 0.4176 - val_acc: 0.8523
    Epoch 4/30
    329/329 [==============================] - 3s 10ms/step - loss: 0.3823 - acc: 0.8638 - val_loss: 0.3829 - val_acc: 0.8643
    Epoch 5/30
    329/329 [==============================] - 3s 9ms/step - loss: 0.3636 - acc: 0.8698 - val_loss: 0.3824 - val_acc: 0.8638
    Epoch 6/30
    329/329 [==============================] - 3s 9ms/step - loss: 0.3494 - acc: 0.8739 - val_loss: 0.4131 - val_acc: 0.8624
    Epoch 7/30
    329/329 [==============================] - 3s 10ms/step - loss: 0.3338 - acc: 0.8787 - val_loss: 0.3614 - val_acc: 0.8739
    Epoch 8/30
    329/329 [==============================] - 5s 14ms/step - loss: 0.3264 - acc: 0.8817 - val_loss: 0.3539 - val_acc: 0.8754
    Epoch 9/30
    329/329 [==============================] - 3s 10ms/step - loss: 0.3138 - acc: 0.8854 - val_loss: 0.3624 - val_acc: 0.8717
    Epoch 10/30
    329/329 [==============================] - 3s 10ms/step - loss: 0.3044 - acc: 0.8896 - val_loss: 0.3557 - val_acc: 0.8731
    Epoch 11/30
    329/329 [==============================] - 4s 11ms/step - loss: 0.2942 - acc: 0.8915 - val_loss: 0.3355 - val_acc: 0.8824
    Epoch 12/30
    329/329 [==============================] - 3s 9ms/step - loss: 0.2868 - acc: 0.8948 - val_loss: 0.3341 - val_acc: 0.8796
    Epoch 13/30
    329/329 [==============================] - 3s 9ms/step - loss: 0.2786 - acc: 0.8977 - val_loss: 0.3282 - val_acc: 0.8853
    Epoch 14/30
    329/329 [==============================] - 3s 10ms/step - loss: 0.2725 - acc: 0.8999 - val_loss: 0.3417 - val_acc: 0.8788
    Epoch 15/30
    329/329 [==============================] - 3s 10ms/step - loss: 0.2683 - acc: 0.9009 - val_loss: 0.3482 - val_acc: 0.8745
    Epoch 16/30
    329/329 [==============================] - 3s 8ms/step - loss: 0.2599 - acc: 0.9045 - val_loss: 0.3405 - val_acc: 0.8781
    Epoch 17/30
    329/329 [==============================] - 3s 9ms/step - loss: 0.2593 - acc: 0.9057 - val_loss: 0.3255 - val_acc: 0.8854
    Epoch 18/30
    329/329 [==============================] - 2s 5ms/step - loss: 0.2489 - acc: 0.9072 - val_loss: 0.3258 - val_acc: 0.8867
    Epoch 19/30
    329/329 [==============================] - 2s 5ms/step - loss: 0.2456 - acc: 0.9100 - val_loss: 0.3385 - val_acc: 0.8822
    Epoch 20/30
    329/329 [==============================] - 1s 4ms/step - loss: 0.2404 - acc: 0.9104 - val_loss: 0.3322 - val_acc: 0.8854
    Epoch 21/30
    329/329 [==============================] - 1s 4ms/step - loss: 0.2364 - acc: 0.9125 - val_loss: 0.3166 - val_acc: 0.8889
    Epoch 22/30
    329/329 [==============================] - 2s 5ms/step - loss: 0.2306 - acc: 0.9159 - val_loss: 0.3410 - val_acc: 0.8818
    Epoch 23/30
    329/329 [==============================] - 1s 4ms/step - loss: 0.2251 - acc: 0.9166 - val_loss: 0.3470 - val_acc: 0.8794
    Epoch 24/30
    329/329 [==============================] - 1s 4ms/step - loss: 0.2235 - acc: 0.9165 - val_loss: 0.3285 - val_acc: 0.8853
    Epoch 25/30
    329/329 [==============================] - 2s 5ms/step - loss: 0.2264 - acc: 0.9156 - val_loss: 0.3250 - val_acc: 0.8883
    Epoch 26/30
    329/329 [==============================] - 1s 5ms/step - loss: 0.2170 - acc: 0.9195 - val_loss: 0.3260 - val_acc: 0.8868
    Epoch 27/30
    329/329 [==============================] - 2s 5ms/step - loss: 0.2090 - acc: 0.9245 - val_loss: 0.3683 - val_acc: 0.8758
    Epoch 28/30
    329/329 [==============================] - 1s 4ms/step - loss: 0.2094 - acc: 0.9223 - val_loss: 0.3362 - val_acc: 0.8827
    Epoch 29/30
    329/329 [==============================] - 1s 4ms/step - loss: 0.2061 - acc: 0.9232 - val_loss: 0.3461 - val_acc: 0.8814
    Epoch 30/30
    329/329 [==============================] - 1s 4ms/step - loss: 0.1978 - acc: 0.9268 - val_loss: 0.3532 - val_acc: 0.8868
    


```python
second_model = Sequential()

second_model.add(Flatten(input_shape=(28,28))) 
second_model.add(Dense(128, activation = 'relu')) # Dense층 추가가
second_model.add(Dense(64, activation = 'relu')) 
second_model.add(Dense(32, activation = 'relu')) 
second_model.add(Dense(10, activation = 'softmax')) 

second_model.compile(optimizer='adam', 
                          loss='categorical_crossentropy', 
                          metrics=['acc']) 

second_history = second_model.fit(x_train, y_train, 
                                    epochs=30, batch_size=128, 
                                    validation_data=(x_val, y_val))
```

    Epoch 1/30
    329/329 [==============================] - 3s 7ms/step - loss: 0.6283 - acc: 0.7854 - val_loss: 0.4367 - val_acc: 0.8508
    Epoch 2/30
    329/329 [==============================] - 2s 6ms/step - loss: 0.4201 - acc: 0.8505 - val_loss: 0.4171 - val_acc: 0.8543
    Epoch 3/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.3761 - acc: 0.8644 - val_loss: 0.3660 - val_acc: 0.8706
    Epoch 4/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.3495 - acc: 0.8713 - val_loss: 0.3704 - val_acc: 0.8668
    Epoch 5/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.3282 - acc: 0.8811 - val_loss: 0.3398 - val_acc: 0.8791
    Epoch 6/30
    329/329 [==============================] - 3s 8ms/step - loss: 0.3113 - acc: 0.8866 - val_loss: 0.3544 - val_acc: 0.8724
    Epoch 7/30
    329/329 [==============================] - 3s 8ms/step - loss: 0.2917 - acc: 0.8917 - val_loss: 0.3348 - val_acc: 0.8788
    Epoch 8/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.2853 - acc: 0.8949 - val_loss: 0.3285 - val_acc: 0.8837
    Epoch 9/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.2711 - acc: 0.8992 - val_loss: 0.3143 - val_acc: 0.8891
    Epoch 10/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.2611 - acc: 0.9031 - val_loss: 0.3342 - val_acc: 0.8791
    Epoch 11/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.2576 - acc: 0.9028 - val_loss: 0.2998 - val_acc: 0.8922
    Epoch 12/30
    329/329 [==============================] - 2s 6ms/step - loss: 0.2449 - acc: 0.9090 - val_loss: 0.3117 - val_acc: 0.8923
    Epoch 13/30
    329/329 [==============================] - 2s 8ms/step - loss: 0.2365 - acc: 0.9117 - val_loss: 0.3193 - val_acc: 0.8894
    Epoch 14/30
    329/329 [==============================] - 2s 5ms/step - loss: 0.2328 - acc: 0.9127 - val_loss: 0.3143 - val_acc: 0.8903
    Epoch 15/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.2271 - acc: 0.9150 - val_loss: 0.3053 - val_acc: 0.8904
    Epoch 16/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.2160 - acc: 0.9182 - val_loss: 0.3085 - val_acc: 0.8943
    Epoch 17/30
    329/329 [==============================] - 3s 8ms/step - loss: 0.2122 - acc: 0.9198 - val_loss: 0.3436 - val_acc: 0.8798
    Epoch 18/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.2070 - acc: 0.9215 - val_loss: 0.3131 - val_acc: 0.8918
    Epoch 19/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.1980 - acc: 0.9261 - val_loss: 0.3241 - val_acc: 0.8921
    Epoch 20/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.1959 - acc: 0.9264 - val_loss: 0.3203 - val_acc: 0.8934
    Epoch 21/30
    329/329 [==============================] - 2s 6ms/step - loss: 0.1899 - acc: 0.9285 - val_loss: 0.3208 - val_acc: 0.8916
    Epoch 22/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.1895 - acc: 0.9289 - val_loss: 0.3190 - val_acc: 0.8948
    Epoch 23/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.1782 - acc: 0.9333 - val_loss: 0.3200 - val_acc: 0.8968
    Epoch 24/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.1811 - acc: 0.9316 - val_loss: 0.3346 - val_acc: 0.8963
    Epoch 25/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.1699 - acc: 0.9356 - val_loss: 0.3327 - val_acc: 0.8950
    Epoch 26/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.1637 - acc: 0.9381 - val_loss: 0.3551 - val_acc: 0.8895
    Epoch 27/30
    329/329 [==============================] - 2s 6ms/step - loss: 0.1632 - acc: 0.9381 - val_loss: 0.3527 - val_acc: 0.8956
    Epoch 28/30
    329/329 [==============================] - 2s 7ms/step - loss: 0.1587 - acc: 0.9407 - val_loss: 0.3638 - val_acc: 0.8881
    Epoch 29/30
    329/329 [==============================] - 3s 9ms/step - loss: 0.1529 - acc: 0.9418 - val_loss: 0.3760 - val_acc: 0.8862
    Epoch 30/30
    329/329 [==============================] - 3s 9ms/step - loss: 0.1498 - acc: 0.9428 - val_loss: 0.3840 - val_acc: 0.8853
    


```python
second_model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_1 (Flatten)         (None, 784)               0         
                                                                     
     dense_3 (Dense)             (None, 128)               100480    
                                                                     
     dense_4 (Dense)             (None, 64)                8256      
                                                                     
     dense_5 (Dense)             (None, 32)                2080      
                                                                     
     dense_6 (Dense)             (None, 10)                330       
                                                                     
    =================================================================
    Total params: 111,146
    Trainable params: 111,146
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# 두 모델의 학습 과정 그려보기
def draw_loss_acc(history1, history2, epochs):
  his_dict_1 = history1.history
  his_dict_2 = history2.history
  keys = list(his_dict_1.keys())

  epochs = range(1, epochs)
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1,1,1)
  # axis선과 ax의 축 레이블 제거
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_color('none')
  ax.spines['left'].set_color('none')
  ax.spines['right'].set_color('none')
  ax.tick_params(labelcolor='w',top=False,bottom=False,left=False,right=False)

  for i in range(len(his_dict_1)):
    temp_ax = fig.add_subplot(2, 2, i+1)
    temp = keys[i%2] # i에 0, 1, 2, 3 순서로 들어감
    val_temp = keys[(i+2)%2 + 2] # 2, 3, 4, 5 순서로 들어감 -> 0, 1, 0, 1 -> 2, 3, 2, 3
    temp_history = his_dict_1 if i < 2 else his_dict_2 
    temp_ax.plot(epochs,temp_history[temp][1:],color='blue',label='train_'+temp)
    temp_ax.plot(epochs,temp_history[val_temp][1:],color='orange',label=val_temp)
    if(i==1 or i==3):  # i가 홀수값일 때
      start,end = temp_ax.get_ylim()
      temp_ax.yaxis.set_ticks(np.arange(np.round(start,2),end,0.01))
    temp_ax.legend()
  ax.set_ylabel('loss',size=20,labelpad=20)
  ax.set_xlabel('Epochs',size=20,labelpad=20)
  plt.tight_layout()
  plt.show()

draw_loss_acc(first_history, second_history, 30)
```


    
![png](04_fashion_mnist_files/04_fashion_mnist_22_0.png)
    



```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
from PIL import Image
import numpy as np
```


```python
img = Image.open('/content/drive/MyDrive/Colab Notebooks/sesac_deeplearning/04_fashion_mnist_img/img02.jpg')
img = img.convert('L')
img = img.resize((28,28))
img = np.array(img)
img = (255-img)/255
plt.imshow(img, cmap='gray')
plt.show()
```


    
![png](04_fashion_mnist_files/04_fashion_mnist_25_0.png)
    



```python
img.shape,x_train.shape
```




    ((28, 28), (42000, 28, 28))




```python
result = first_model.predict(img.reshape(-1,28,28))
```

    1/1 [==============================] - 0s 154ms/step
    


```python
result.shape
```




    (1, 10)




```python
np.argmax(np.round(result,2))
```




    8




```python
class_names[np.argmax(np.round(result,2))]
```




    'Bag'




```python
img = Image.open('/content/drive/MyDrive/Colab Notebooks/sesac_deeplearning/04_fashion_mnist_img/img03.jpg')
img = img.convert('L')
img = img.resize((28,28))
img = np.array(img)
img = (255-img)/255
plt.imshow(img, cmap='gray')
plt.show()
```


    
![png](04_fashion_mnist_files/04_fashion_mnist_31_0.png)
    



```python
result = first_model.predict(img.reshape(-1,28,28))
np.argmax(np.round(result,2))
class_names[np.argmax(np.round(result,2))]
```

    1/1 [==============================] - 0s 17ms/step
    




    'Sandal'




```python
img = Image.open('/content/drive/MyDrive/Colab Notebooks/sesac_deeplearning/04_fashion_mnist_img/img04.jpg')
img = img.convert('L')
img = img.resize((28,28))
img = np.array(img)
img = (255-img)/255
plt.imshow(img, cmap='gray')
plt.show()
```


    
![png](04_fashion_mnist_files/04_fashion_mnist_33_0.png)
    



```python
result = first_model.predict(img.reshape(-1,28,28))
np.argmax(np.round(result,2))
class_names[np.argmax(np.round(result,2))]
```

    1/1 [==============================] - 0s 45ms/step
    




    'Shirt'




```python

```
