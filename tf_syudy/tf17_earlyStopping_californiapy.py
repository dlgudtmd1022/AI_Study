import numpy as py
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from keras.callbacks import EarlyStopping

# 1. 데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape)  # (20640, 8)
print(y.shape) # (20640,)
print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test =train_test_split(
    x, y, 
    train_size=0.6,
    test_size=0.2, 
    random_state=72, 
    shuffle=True,
)

print(x_train.shape)     # (14447, 8)
print(y_train.shape)     # (14447,)
print(x_test.shape)      # (6193, 8)
print(y_test.shape)      # (6193,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성

model = Sequential()
model.add(Dense(128, input_dim = 8))
model.add(Dense(10))
model.add(Dense(24))
model.add(Dense(1))

# 3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer='adam')

# earplyStopping 적용

earlyStopping = EarlyStopping(
    monitor='val_loss', 
    patience=50,
    mode='min',
    verbose= 1,
    restore_best_weights=True
)
hist = model.fit(x_train, y_train,
                 epochs=5000, 
                 batch_size=123, 
                 validation_split=0.2, 
                 callbacks=[earlyStopping])

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

# r2 score 결정계수 #
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label = 'val_loss')
plt.title('Loss & Val_Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()