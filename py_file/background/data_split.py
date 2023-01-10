import numpy as np

def minMaxScaler(X, x_max, x_min):
  X = X.copy()
  for feature in x_max.keys():
    X[feature] = (X[feature] - x_min[feature]) / (x_max[feature] - x_min[feature])
  return X

def normalization(X_train, X_test):
  features = list(X_train.columns)
  x_max = {}
  x_min = {}
  for feature in features:
    x_max[feature] = X_train[feature].max()
    x_min[feature] = X_train[feature].min()
  X_train = minMaxScaler(X_train, x_max, x_min)
  X_test = minMaxScaler(X_test, x_max, x_min)
  return X_train, X_test

# 학습/테스트 데이터 뒤섞기
def df_shuffle(df):
  return df.sample(frac=1).reset_index(drop=True)

# 특성, 타겟 나누기
def divide_xy(df):
  target = 'y'
  features = list(df.columns)
  features.remove(target)

  X = df[features]
  y = np.array(df[target]).reshape(-1, 1)
  return X, y

# 학습/테스트 데이터 분리하기
def train_test_divide(X, y, test_size=0.2):  # X: pandas dataframe, y: numpy array
  length = len(y)
  test_index = int(length * test_size)

  X_test = X[:test_index]
  y_test = y[:test_index]

  X_train = X[test_index:]
  y_train = y[test_index:]

  # train 데이터에 맞춰서 normalization(minMaxScaler 이용)
  X_train, X_test = normalization(X_train, X_test)

  return X_train, y_train, X_test, y_test

# 위의 함수 통합
def train_test_split(df, shuffle=True, test_size=0.2):
  if shuffle:
    df = df_shuffle(df)

  X, y = divide_xy(df)
  return train_test_divide(X, y, test_size=test_size)

# 미니배치 설정
def split_mini_batch(X, y, n): # train, test 들어올 예정
  # 4개씩 미니배치 설정. 나머지는 버리기
  length = len(y)
  num = length // n # 미니배치 개수
  X_batch_list = []
  y_batch_list = []
  for i in range(num):
    i = i * n
    # 비복원 추출. 데이터가 적기 때문에 겹치지 않는 게 나을 듯.
    X_batch_list.append(X[i:i+n]) # index: 0~4, 4~8, 8~12, ...
    y_batch_list.append(y[i:i+n])
  return X_batch_list, y_batch_list

def train_test_mini_batch(X_train, y_train, X_test, y_test, n):
  train_X_batch_list, train_y_batch_list = split_mini_batch(X_train, y_train, n) # train 데이터
  test_X_batch_list, test_y_batch_list = split_mini_batch(X_test, y_test, n) # test 데이터
  return train_X_batch_list, train_y_batch_list, test_X_batch_list, test_y_batch_list