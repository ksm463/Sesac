### 컨볼루션층과 풀링층
* 완전연결층은 1차원 배열의 형태라 공간 정보를 손실하게 된다.
* 컨볼루션층은 이미지 픽셀 사이의 관계를 고려하기 때문에 공간정보를 유지한다.
* 필터(filter=kernel)를 정의하여 사용할 수 있음.
* 필터는 짝수 개면 비대칭이 되므로, 홀수 개로 지정하여야 한다.


```python
from keras.datasets import fashion_mnist
```


```python
# 데이터 다운로드
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
```


```python
plt.imshow(x_train[0],cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7faa7d395d00>




    
![png](08_cnn_filter_files/08_cnn_filter_4_1.png)
    



```python
import numpy as np
import cv2
```


```python
# 가로선을 추출하기 위한 필터
h_filter = np.array([[1.,2.,1.],
                        [0.,0.,0.],
                        [-1.,-2.,-1.]])

# 세로선을 추출하기 위한 필터터
v_filter = np.array([[1.,0.,-1.],
                        [2.,0.,-2.],
                        [1.,0.,-1.]])
```


```python
# 계산의 편의를 위해 이미지를 (27, 27)로 줄임.
test_image = cv2.resize(x_train[0],(27,27))
test_image.shape
```




    (27, 27)




```python
# 사이즈 변환
image_size = test_image.shape[0]
output_size = int((image_size - 3)/1 + 1) # 필터가 세개이므로 3을 빼줌. 그 다음에 strider + 1로 나눔
output_size
```




    25




```python
filter_size = 3

def get_filtered_image(filter):
  filtered_image = np.zeros((output_size,output_size))
  for i in range(output_size):
    for j in range(output_size):
      indice_image = test_image[i:(i+filter_size),j:(j+filter_size)] * filter #  i*j에 filter_size(=3)을을 더함.
      indice_sum = np.sum(indice_image)

      if(indice_sum>255):
        indice_sum =255
      filtered_image[i,j] = indice_sum
  return filtered_image
```


```python
h_filtered_image = get_filtered_image(h_filter)
v_filtered_image = get_filtered_image(v_filter)
```


```python
plt.subplot(1,2,1)
plt.title('vertical')
plt.imshow(v_filtered_image,cmap='gray')

plt.subplot(1,2,2)
plt.title('horizontal')
plt.imshow(h_filtered_image,cmap='gray')
plt.show()
```


    
![png](08_cnn_filter_files/08_cnn_filter_11_0.png)
    



```python
# 수직 + 수평 최종 결과
sobel_image = np.sqrt(np.square(h_filtered_image) + np.square(v_filtered_image))
plt.imshow(sobel_image,cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7faa6f66b610>




    
![png](08_cnn_filter_files/08_cnn_filter_12_1.png)
    


* 스트라이드 : 필터가 움직이는 크기를 말한다. 예를 들어 데이터셋 5X5에서 3X3 필터가 stride 1이면 3X3가지, stride가 2이면 2X2가 나온다.
* 패딩 : 가장자리에 칸을 추가하여 입력 데이터와 동일한 사이즈를 얻기 위해 사용

### 풀링 연산 알아보기
* 평균 풀링 : 평균값 이용
* 최대 풀링 : 최댓값 이용
* 해당 윈도우에서 평균or최댓값을 특징값으로 사용함
* 최대 풀링은 신경망에 이동 불변성을 가지게 해주고, 모델 파라미터 수를 줄여줌. 1X1스트라이드 사용을 권장함.
