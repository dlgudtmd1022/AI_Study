import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터

x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2.1, 3.1, 4.1, 5.1, 6, 7, 8.1, 9.2, 10.5]])
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

print(x.shape)
print(y.shape)

x = x.transpose()   # x = x.T
print(x.shape)

# 2. 모델구성

model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(8))
model.add(Dense(13))
model.add(Dense(11))
model.add(Dense(4))
model.add(Dense(1))

# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000 ,batch_size= 4)

# 4. 평가, 예측

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[10, 10.5]])
print('10과 10.5의 예측값 : ', result)