import numpy as np

# 1. 데이터

x = np.array([1, 2, 3, 5, 4])
y = np.array([1, 2, 3, 4, 5])

# 2. 모델구성

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련

model.compile(loss="mae", optimizer="adam")

model.fit(x, y, epochs=1000)


# 4. 평가, 예측

loss = model.evaluate(x, y) #  0.40174221992492676
print('loss : ', loss)

result = model.predict([6]) # [[5.9919367]]
print('result : ', result)