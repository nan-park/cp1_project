# numpy, pandas, csv, matplotlib 라이브러리만 사용 가능
import numpy as np
import pandas as pd
from background.data_load import data_load
from background.data_split import train_test_split, split_mini_batch
from background.sequential import Sequential, Input, Dense
from background.evaluate import evaluate

def main():

    # csv 데이터 불러오기
    df = data_load()

    # train, test 데이터로 나누기
    X_train, y_train, X_test, y_test = train_test_split(df)
    # 미니배치(하나에 4개 데이터)로 나누기
    X_batch_list, y_batch_list = split_mini_batch(X_train, y_train) # train 데이터만

    # 모델 만들기(입력층, 은닉층, 출력층)
    model = Sequential([Input(8), Dense(16), Dense(32), Dense(1, activation='sigmoid')])

    # 정확도, 손실함수 측정하기
    accuracy, cross_entropy = evaluate(X_batch_list, y_batch_list, model)
    # 결과 출력하기
    print(f"[Epoch 1] TrainData - Loss = {round(cross_entropy, 3)}, Accuracy = {round(accuracy, 3)}")

main()