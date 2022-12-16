## CatsAndDogs
* 강아지와 고양이 이미지를 이용한 실습


```python
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import zipfile
```


```python
from google.colab import drive
drive.mount("/content/drive")
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
path = '/content/drive/MyDrive/Colab Notebooks/sesac_deeplearning/cats_and_dogs.zip'
zip_ref = zipfile.ZipFile(path,'r')
zip_ref.extractall('dataset/')
zip_ref.close()
```


```python
image_gen = ImageDataGenerator(rescale=(1/255.))
train_dir = '/content/dataset/cats_and_dogs_filtered/train'
valid_dir = '/content/dataset/cats_and_dogs_filtered/validation'
train_gen = image_gen.flow_from_directory(train_dir,
                                                          target_size=(224,224),
                                                          batch_size=32, 
                                                          classes=['cats','dogs'], 
                                                          class_mode='binary',  
                                                          seed=2020)

valid_gen = image_gen.flow_from_directory(valid_dir,
                                                          target_size=(224,224),
                                                          batch_size=32, 
                                                          classes=['cats','dogs'], 
                                                          class_mode='binary',  
                                                          seed=2020)
```

    Found 2000 images belonging to 2 classes.
    Found 1000 images belonging to 2 classes.
    


```python
class_labels=['cats','dogs']
batch = next(train_gen)
images, labels = batch[0],batch[1]

plt.figure(figsize=(16,8))
for i in range(32):
  ax = plt.subplot(4,8,i+1)
  plt.imshow(images[i])
  plt.title(class_labels[labels[i].astype(int)])
  plt.axis('off')
plt.tight_layout()
plt.show()
```


    
![png](17_cnn_CatsAndDogs_files/17_cnn_CatsAndDogs_5_0.png)
    



```python
images.shape
```




    (32, 224, 224, 3)




```python
from keras import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

def build_model():
  model = Sequential([
      BatchNormalization(),
      Conv2D(32,(3,3),padding='same',activation='relu'),
      MaxPooling2D((2,2)),

      BatchNormalization(),
      Conv2D(64,(3,3),padding='same',activation='relu'),
      MaxPooling2D((2,2)),

      BatchNormalization(),
      Conv2D(128,(3,3),padding='same',activation='relu'),
      MaxPooling2D((2,2)),

      Flatten(),
      Dense(256,activation='relu'),
      Dropout(0.3),
      Dense(1,activation='sigmoid'),
  ])
  return model
```


```python
model = build_model()
```


```python
model.compile(optimizer=Adam(0.001),loss=BinaryCrossentropy(),metrics=['acc'])
history = model.fit(train_gen,validation_data=valid_gen,epochs=20)
```

    Epoch 1/20
    63/63 [==============================] - 14s 189ms/step - loss: 7.1242 - acc: 0.5330 - val_loss: 0.6887 - val_acc: 0.5930
    Epoch 2/20
    63/63 [==============================] - 11s 171ms/step - loss: 0.6544 - acc: 0.6050 - val_loss: 0.6836 - val_acc: 0.5790
    Epoch 3/20
    63/63 [==============================] - 11s 170ms/step - loss: 0.6279 - acc: 0.6390 - val_loss: 0.6743 - val_acc: 0.5750
    Epoch 4/20
    63/63 [==============================] - 11s 170ms/step - loss: 0.6157 - acc: 0.6545 - val_loss: 0.6748 - val_acc: 0.5720
    Epoch 5/20
    63/63 [==============================] - 14s 221ms/step - loss: 0.5988 - acc: 0.6665 - val_loss: 0.6319 - val_acc: 0.6480
    Epoch 6/20
    63/63 [==============================] - 13s 210ms/step - loss: 0.5706 - acc: 0.6955 - val_loss: 0.6056 - val_acc: 0.6610
    Epoch 7/20
    63/63 [==============================] - 15s 241ms/step - loss: 0.5375 - acc: 0.7115 - val_loss: 0.5861 - val_acc: 0.6700
    Epoch 8/20
    63/63 [==============================] - 13s 200ms/step - loss: 0.5178 - acc: 0.7230 - val_loss: 0.6501 - val_acc: 0.6570
    Epoch 9/20
    63/63 [==============================] - 11s 168ms/step - loss: 0.4845 - acc: 0.7530 - val_loss: 0.5640 - val_acc: 0.7000
    Epoch 10/20
    63/63 [==============================] - 11s 167ms/step - loss: 0.4808 - acc: 0.7520 - val_loss: 0.5579 - val_acc: 0.7130
    Epoch 11/20
    63/63 [==============================] - 11s 169ms/step - loss: 0.4450 - acc: 0.7615 - val_loss: 0.6506 - val_acc: 0.6720
    Epoch 12/20
    63/63 [==============================] - 11s 167ms/step - loss: 0.4294 - acc: 0.7740 - val_loss: 0.7200 - val_acc: 0.6850
    Epoch 13/20
    63/63 [==============================] - 11s 179ms/step - loss: 0.4301 - acc: 0.7845 - val_loss: 0.6644 - val_acc: 0.6740
    Epoch 14/20
    63/63 [==============================] - 11s 174ms/step - loss: 0.3827 - acc: 0.8005 - val_loss: 0.6203 - val_acc: 0.6890
    Epoch 15/20
    63/63 [==============================] - 11s 168ms/step - loss: 0.3294 - acc: 0.8335 - val_loss: 0.5823 - val_acc: 0.6840
    Epoch 16/20
    63/63 [==============================] - 15s 234ms/step - loss: 0.3610 - acc: 0.8260 - val_loss: 0.6217 - val_acc: 0.7050
    Epoch 17/20
    63/63 [==============================] - 11s 167ms/step - loss: 0.3446 - acc: 0.8160 - val_loss: 0.6852 - val_acc: 0.6740
    Epoch 18/20
    63/63 [==============================] - 11s 167ms/step - loss: 0.3042 - acc: 0.8600 - val_loss: 0.7751 - val_acc: 0.6860
    Epoch 19/20
    63/63 [==============================] - 11s 167ms/step - loss: 0.3159 - acc: 0.8475 - val_loss: 0.7343 - val_acc: 0.6910
    Epoch 20/20
    63/63 [==============================] - 11s 169ms/step - loss: 0.2872 - acc: 0.8525 - val_loss: 0.7057 - val_acc: 0.7140
    


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
```


```python
plot_loss_acc(history,20)
```


    
![png](17_cnn_CatsAndDogs_files/17_cnn_CatsAndDogs_11_0.png)
    



```python
image_gen = ImageDataGenerator(rescale=(1/255.), horizontal_flip=True, zoom_range=0.2, rotation_range=35)
train_dir = '/content/dataset/cats_and_dogs_filtered/train'
valid_dir = '/content/dataset/cats_and_dogs_filtered/validation'
train_gen_aug = image_gen.flow_from_directory(train_dir,
                                                          target_size=(224,224),
                                                          batch_size=32, 
                                                          classes=['cats','dogs'], 
                                                          class_mode='binary',  
                                                          seed=2020)

valid_gen_aug = image_gen.flow_from_directory(valid_dir,
                                                          target_size=(224,224),
                                                          batch_size=32, 
                                                          classes=['cats','dogs'], 
                                                          class_mode='binary',  
                                                          seed=2020)
```

    Found 2000 images belonging to 2 classes.
    Found 1000 images belonging to 2 classes.
    


```python
model_aug = build_model()
model_aug.compile(optimizer=Adam(0.001),loss=BinaryCrossentropy(),metrics=['acc'])
history_aug = model_aug.fit(train_gen_aug,validation_data=valid_gen_aug,epochs=20)
```

    Epoch 1/20
    63/63 [==============================] - 38s 587ms/step - loss: 4.4823 - acc: 0.5235 - val_loss: 0.6911 - val_acc: 0.5040
    Epoch 2/20
    63/63 [==============================] - 36s 577ms/step - loss: 0.6575 - acc: 0.6090 - val_loss: 0.6842 - val_acc: 0.5250
    Epoch 3/20
    63/63 [==============================] - 36s 573ms/step - loss: 0.6329 - acc: 0.6415 - val_loss: 0.6666 - val_acc: 0.6000
    Epoch 4/20
    63/63 [==============================] - 43s 690ms/step - loss: 0.6192 - acc: 0.6560 - val_loss: 0.6440 - val_acc: 0.6330
    Epoch 5/20
    63/63 [==============================] - 39s 618ms/step - loss: 0.5996 - acc: 0.6590 - val_loss: 0.6389 - val_acc: 0.6290
    Epoch 6/20
    63/63 [==============================] - 38s 607ms/step - loss: 0.6070 - acc: 0.6605 - val_loss: 0.6210 - val_acc: 0.6360
    Epoch 7/20
    63/63 [==============================] - 36s 576ms/step - loss: 0.5959 - acc: 0.6885 - val_loss: 0.5941 - val_acc: 0.6630
    Epoch 8/20
    63/63 [==============================] - 37s 594ms/step - loss: 0.5956 - acc: 0.6685 - val_loss: 0.6083 - val_acc: 0.6660
    Epoch 9/20
    63/63 [==============================] - 37s 588ms/step - loss: 0.5840 - acc: 0.6910 - val_loss: 0.6401 - val_acc: 0.6310
    Epoch 10/20
    63/63 [==============================] - 38s 599ms/step - loss: 0.5757 - acc: 0.6785 - val_loss: 0.5831 - val_acc: 0.6910
    Epoch 11/20
    63/63 [==============================] - 36s 575ms/step - loss: 0.5627 - acc: 0.7075 - val_loss: 0.6065 - val_acc: 0.6710
    Epoch 12/20
    63/63 [==============================] - 36s 574ms/step - loss: 0.5695 - acc: 0.6885 - val_loss: 0.5830 - val_acc: 0.6970
    Epoch 13/20
    63/63 [==============================] - 36s 576ms/step - loss: 0.5762 - acc: 0.7020 - val_loss: 0.5840 - val_acc: 0.6820
    Epoch 14/20
    63/63 [==============================] - 36s 572ms/step - loss: 0.5563 - acc: 0.6935 - val_loss: 0.6023 - val_acc: 0.6640
    Epoch 15/20
    63/63 [==============================] - 36s 575ms/step - loss: 0.5619 - acc: 0.6870 - val_loss: 0.5911 - val_acc: 0.6590
    Epoch 16/20
    63/63 [==============================] - 37s 591ms/step - loss: 0.5740 - acc: 0.6710 - val_loss: 0.5882 - val_acc: 0.6900
    Epoch 17/20
    63/63 [==============================] - 36s 575ms/step - loss: 0.5553 - acc: 0.7045 - val_loss: 0.5769 - val_acc: 0.6710
    Epoch 18/20
    63/63 [==============================] - 36s 574ms/step - loss: 0.5466 - acc: 0.7260 - val_loss: 0.6171 - val_acc: 0.6860
    Epoch 19/20
    63/63 [==============================] - 36s 573ms/step - loss: 0.5465 - acc: 0.7170 - val_loss: 0.5709 - val_acc: 0.7040
    Epoch 20/20
    63/63 [==============================] - 36s 570ms/step - loss: 0.5493 - acc: 0.7085 - val_loss: 0.5507 - val_acc: 0.7240
    


```python
plot_loss_acc(history_aug,20)
```


    
![png](17_cnn_CatsAndDogs_files/17_cnn_CatsAndDogs_14_0.png)
    



```python
from keras.applications import ResNet50V2
pre_trained_base = ResNet50V2(include_top=False, 
                               weights='imagenet',
                               input_shape=[224, 224, 3])
pre_trained_base.trainable = False
def build_trainsfer_model():

    model = tf.keras.Sequential([

        # Pre-trained Base 
        pre_trained_base,
        
        # Classifier 출력층 
      Flatten(),
      Dense(256,activation='relu'),
      Dropout(0.3),
      Dense(1,activation='sigmoid'),
    ])

    return model
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
    94668760/94668760 [==============================] - 1s 0us/step
    


```python
tc_model = build_trainsfer_model()
tc_model.compile(optimizer=Adam(0.001),loss=BinaryCrossentropy(),metrics=['acc'])
history_t = tc_model.fit(train_gen_aug,validation_data=valid_gen_aug,epochs=20)
```

    Epoch 1/20
    63/63 [==============================] - 41s 615ms/step - loss: 1.9358 - acc: 0.9325 - val_loss: 1.3339 - val_acc: 0.9550
    Epoch 2/20
    63/63 [==============================] - 38s 601ms/step - loss: 1.0656 - acc: 0.9605 - val_loss: 0.6684 - val_acc: 0.9720
    Epoch 3/20
    63/63 [==============================] - 39s 614ms/step - loss: 0.4522 - acc: 0.9725 - val_loss: 0.4224 - val_acc: 0.9740
    Epoch 4/20
    63/63 [==============================] - 37s 594ms/step - loss: 0.2019 - acc: 0.9790 - val_loss: 0.3039 - val_acc: 0.9640
    Epoch 5/20
    63/63 [==============================] - 37s 589ms/step - loss: 0.1355 - acc: 0.9780 - val_loss: 0.2052 - val_acc: 0.9710
    Epoch 6/20
    63/63 [==============================] - 37s 595ms/step - loss: 0.0922 - acc: 0.9815 - val_loss: 0.1635 - val_acc: 0.9650
    Epoch 7/20
    63/63 [==============================] - 37s 592ms/step - loss: 0.0811 - acc: 0.9835 - val_loss: 0.1686 - val_acc: 0.9660
    Epoch 8/20
    63/63 [==============================] - 38s 597ms/step - loss: 0.0742 - acc: 0.9880 - val_loss: 0.1315 - val_acc: 0.9760
    Epoch 9/20
    63/63 [==============================] - 45s 717ms/step - loss: 0.0337 - acc: 0.9870 - val_loss: 0.2035 - val_acc: 0.9680
    Epoch 10/20
    63/63 [==============================] - 38s 600ms/step - loss: 0.0952 - acc: 0.9805 - val_loss: 0.1748 - val_acc: 0.9680
    Epoch 11/20
    63/63 [==============================] - 38s 613ms/step - loss: 0.1170 - acc: 0.9840 - val_loss: 0.2379 - val_acc: 0.9700
    Epoch 12/20
    63/63 [==============================] - 37s 594ms/step - loss: 0.1105 - acc: 0.9870 - val_loss: 0.1558 - val_acc: 0.9740
    Epoch 13/20
    63/63 [==============================] - 37s 596ms/step - loss: 0.0446 - acc: 0.9895 - val_loss: 0.1487 - val_acc: 0.9740
    Epoch 14/20
    63/63 [==============================] - 39s 615ms/step - loss: 0.0372 - acc: 0.9935 - val_loss: 0.0970 - val_acc: 0.9700
    Epoch 15/20
    63/63 [==============================] - 37s 590ms/step - loss: 0.0394 - acc: 0.9905 - val_loss: 0.2279 - val_acc: 0.9670
    Epoch 16/20
    63/63 [==============================] - 37s 591ms/step - loss: 0.0411 - acc: 0.9905 - val_loss: 0.1350 - val_acc: 0.9690
    Epoch 17/20
    63/63 [==============================] - 37s 590ms/step - loss: 0.0496 - acc: 0.9860 - val_loss: 0.2290 - val_acc: 0.9660
    Epoch 18/20
    63/63 [==============================] - 37s 593ms/step - loss: 0.0713 - acc: 0.9865 - val_loss: 0.1853 - val_acc: 0.9710
    Epoch 19/20
    63/63 [==============================] - 37s 592ms/step - loss: 0.0573 - acc: 0.9885 - val_loss: 0.1257 - val_acc: 0.9730
    Epoch 20/20
    63/63 [==============================] - 38s 610ms/step - loss: 0.0309 - acc: 0.9910 - val_loss: 0.1306 - val_acc: 0.9700
    


```python
plot_loss_acc(history_t,20)
```


    
![png](17_cnn_CatsAndDogs_files/17_cnn_CatsAndDogs_17_0.png)
    



```python

```
