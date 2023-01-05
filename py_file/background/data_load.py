import pandas as pd

# 데이터 불러오기
def data_load():
  df = pd.read_csv('./binary_dataset.csv')
  return df