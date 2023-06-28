from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from keras.utils import to_categorical
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from keras.callbacks import EarlyStopping

# 1. 데이터

datasets = load_wine()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)         # (178, 13) / (178,)
print(datasets.feature_names)   # ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
print(datasets.DESCR) #     - class:- class_0   /  - class_1   /  - class_2

# one-hot encoding
y = to_categorical(y)
print(y.shape) # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.6,
    random_state=72,
    shuffle=True
)

print(x_train.shape, x_test.shape) # (124, 13) (54, 13)
print(y_train.shape, y_test.shape) # (124, 3) (54, 3)

# Scaler 적용

scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성

model = Sequential()
model.add(Dense(32, input_dim = 13))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['mse', 'accuracy']) 
start_time = time.time()
earlyStopping = EarlyStopping(
    monitor= 'val_loss',
    mode='min',
    patience= 50,
    verbose= 1,
    restore_best_weights=True
)
model.fit(x_train, y_train,
          validation_split=0.2 ,epochs=5000,batch_size=32, callbacks=[earlyStopping])
end_time = time.time() - start_time


# 4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)  # loss :  0.991733968257904
print('mse : ', mse)  # mse :  0.09521209448575974
print('accuracy : ', accuracy) # accuracy :  0.8333333134651184
print('걸린시간 : ', end_time) # 걸린시간 :  8.881948709487915

'''
1. StandardScaler

loss :  0.18480417132377625
mse :  0.019285330548882484
accuracy :  0.9629629850387573

2. MinMax

loss :  0.27250906825065613
mse :  0.020167935639619827
accuracy :  0.9629629850387573

3. MaxAbsScaler

loss :  0.42191657423973083
mse :  0.035615868866443634
accuracy :  0.9444444179534912

4. RobustScaler

loss :  0.21469296514987946
mse :  0.012475820258259773
accuracy :  0.9814814925193787

earlyStopping

Epoch 59: early stopping
3/3 [==============================] - 0s 3ms/step - loss: 0.0376 - mse: 0.0061 - accuracy: 0.9861
loss :  0.03762665018439293
mse :  0.006145365536212921
accuracy :  0.9861111044883728
걸린시간 :  4.754414081573486

'''