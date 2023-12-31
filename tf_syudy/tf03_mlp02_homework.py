import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 1, 1, 2, 1.1, 1.2, 1.4, 1.5, 1.6],
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

print(x.shape)
x = x.T
print(x.shape)
# 모델구성부터 평가예측까지 완성하시오

model = Sequential()
model.add(Dense(10, input_dim = 3))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 1000, batch_size= 8)

# 예측 [[10, 1.6, 1]]

loss = model.evaluate(x, y) # 1.1630787412286736e-05
print('loss : ', loss)

result = model.predict([[10, 1.6, 1]])
print('result : ', result)  # [[19.999617]]