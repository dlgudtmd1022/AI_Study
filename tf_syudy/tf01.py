# 1. 데이터

import numpy as np

x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 2. 모델구성

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim = 1))  # 입력층
model.add(Dense(6))                 # 히든레이어 1
model.add(Dense(8))                 # 히든레이어 2
model.add(Dense(4))                 # 히든레이어 3
model.add(Dense(2))                 # 히든레이어 4
model.add(Dense(1))                 # 출력층

# 3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs=100)

# 4. 평가, 예측

loss = model.evaluate(x, y) # 0.002855087863281369
print('loss : ', loss)

result = model.predict([4]) #  [[3.9933155]]
print('result : ', result)