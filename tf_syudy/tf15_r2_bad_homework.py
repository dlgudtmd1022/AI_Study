# [실습]
# 1. r2 score를 음수가 아닌 0.5이하로 만들기
# 2. 데이터는 건들지 않기
# 3. 레이어는 인풋, 아웃풋 포함 7개 이상
# 4. batch_size=1 로 고정
# 5. 히든레이어의 노트(뉴런) 갯수는 10개 이상 100개 이하
# 6. train_size = 0.7 고정
# 7. epochs = 100 이상
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras import optimizers

# 1. 테이터

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

x_train, x_test, y_train, y_test =train_test_split(
    x, y, 
    train_size=0.7,
    random_state=123, 
    shuffle=False,
)

# 2. 모델구성

model = Sequential()
model.add(Dense(14, input_dim = 1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss = 'mae', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1)

#4. 평가, 예측


# r2 score 결정계수 #
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)