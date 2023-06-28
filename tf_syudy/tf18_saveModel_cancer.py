import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

# 1. 데이터

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

print(x.shape)                    # (569, 30)
print(y.shape)                    # (569, )
print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size= 0.3,
    random_state=1234,
    shuffle=True
)

print(x_train.shape)    # (398, 30)
print(y_train.shape)    # (398,) 
print(x_test.shape)     # (171, 30)  
print(y_test.shape)     # (171,)

# Scaler 적용

# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성

model = Sequential()
model.add(Dense(68,input_dim = 30)) # activatiom = linear가 default
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1, activation='sigmoid')) # 이진분류는 마지막 아웃풋 레이어에 무조건 sigmoid 함수 사용

# 3. 컴파일, 훈련

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=123)

# model 저장
model.save('./_save/tf18_cancer.h5')

# 4. 평가, 예측

loss, mse, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mse : ', mse)
print('accuracy : ', accuracy)

'''
# 실습 : accuracy_score를 출력하기 #
y_predict = model.predict(x_test)
# print(y_predict)

# ~~~ 뭘 넣어야 할까 ~~~~
# y_predict = np.where(y_predict > 0.5, 1, 0)
y_predict = np.round(y_predict)
print(y_predict)
# acc_score = accuracy_score(y_test,y_predict)
# print('acc_score : ', acc_score)
'''

'''
1. StandardScaler
loss :  0.324706107378006
mse :  0.02930193394422531
accuracy :  0.9649122953414917

2. MinMax
loss :  0.30081862211227417
mse :  0.044959764927625656
accuracy :  0.9473684430122375

3. MaxAbsScaler
loss :  0.23245634138584137
mse :  0.043360527604818344
accuracy :  0.9415204524993896

4. RobustScaler
loss :  0.3232281506061554
mse :  0.028468361124396324
accuracy :  0.9649122953414917
'''