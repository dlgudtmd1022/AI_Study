from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time

# 1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# 시각화

# plt.imshow(x_train[0])
# plt.show()

# scaling (이미지 0 ~ 255 => 0 ~ 1 범위로 만들어 줌)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/510
x_test = x_test/510

# 2. 모델구성

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=(32, 32, 3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

earlyStopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=20,
    restore_best_weights=True
)
start_time = time.time()

model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=40,
          callbacks=[earlyStopping],
          batch_size=64
)

end_time = time.time() - start_time

# 4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc : ', acc)
print('걸린시간 : ', end_time)


# loss : 4.607812881469727
# acc :  0.011300000362098217
# 걸린시간 :  237.39023447036743

# scaling 이후
# loss : 2.7338016033172607
# acc :  0.32350000739097595
# 걸린시간 :  243.85253882408142

# 최종
# loss : 2.2680647373199463
# acc :  0.43470001220703125
# 걸린시간 :  2632.3741693496704