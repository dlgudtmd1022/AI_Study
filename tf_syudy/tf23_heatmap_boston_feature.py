import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time, datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns           # pip install seaborn

# 1. 데이터
path = './data/boston/'
datasets = pd.read_csv(path + 'Boston_house.csv')
print(datasets.shape)   # (506, 14)
print(datasets.columns) # Index(['AGE', 'B', 'RM', 'CRIM', 'DIS', 'INDUS', 'LSTAT', 'NOX', 'PTRATIO','RAD', 'ZN', 'TAX', 'CHAS', 'Target'], dtype='object')

'''
# 시각화, 상관관계

sns.set(font_scale = 1.2)
sns.set(rc={'figure.figsize': (9, 6)})
sns.heatmap(
    data = datasets.corr(), # 상관관계
    square = True,          # 정사각형으로 view
    annot = True,           # 각 cell의 값 표기
    cbar = True,            # color bar 표기
)
plt.show()
'''

# x, y 데이터 분리

x = datasets.drop(['Target', 'LSTAT'], axis = 1)
y = datasets.Target

print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.6,
    random_state=72,
    shuffle=True
)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성

model = Sequential()
model.add(Dense(64, input_dim = 13))
model.add(Dense(126))
model.add(Dense(19))
model.add(Dense(19))
model.add(Dense(19))
model.add(Dense(1))

# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

earlyStopping = EarlyStopping(
    monitor='val_loss',
    mode='miin',
    verbose=1,
    patience=50,
    restore_best_weights=True
)

date = datetime.datetime.now()
date = date.strftime("%m%d_%h%m")
filepath = './_mcp/'
filename = '{epoch:04d}{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    verbose=1,
    filepath="".join([filepath, 'ft22_boston', date, '_', filename])
)

model.fit(
    x_train, y_train,
    validation_split=0.2,
    callbacks=[earlyStopping, mcp],
    verbose=1,
    epochs=5000,
    batch_size=128
)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ',r2)

# drop ' LSTAT' r2