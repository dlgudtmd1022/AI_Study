import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from keras.callbacks import EarlyStopping

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)         # (150, 4) (150,)
print(datasets.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']   
print(datasets.DESCR)

# 원한잇코딩 one-hot encoding #

from keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.6,
    test_size=0.2,
    random_state=72,
    shuffle=True
)

# scaler

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
'''
# 2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim = 4))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(3, activation='softmax'))
'''
model = load_model('./_save/tf18_iris.h5')

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['mse', 'accuracy']) 
start_time = time.time()
earlyStopping = EarlyStopping(
    monitor='val_loss',
    mode= 'min',
    patience= 50,
    verbose= 1,
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, 
          validation_split=0.2, epochs=5000, batch_size=32, callbacks=[earlyStopping])
end_time = time.time() - start_time
print('걸린시간 : ', end_time)

# 4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss) #  loss :  0.06169286370277405
print('mse : ', mse) #  mse :  0.014008337631821632
print('accuracy : ', accuracy) # accuracy :  0.9555555582046509

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

'''
earlystopping
uracy: 0.9444
Epoch 127: early stopping
걸린시간 :  8.170478820800781
1/1 [==============================] - 0s 31ms/step - loss: 0.0185 - mse: 9.5430e-04 - accuracy: 1.0000
loss :  0.018497709184885025
mse :  0.0009542988264001906
accuracy :  1.0
'''