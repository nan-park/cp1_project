# numpy, pandas, csv, matplotlib 라이브러리만 사용 가능
import numpy as np
import pandas as pd
from background.data_load import data_load
from background.data_split import train_test_split, train_test_mini_batch
from background.sequential import Sequential, Input, Dense
from background.evaluate import evaluate_epoch


def main():

    # csv 데이터 불러오기
    df = data_load()

    # train, test 데이터로 나누기
    X_train, y_train, X_test, y_test = train_test_split(df)
    # 미니배치(하나에 4개 데이터)로 나누기
    X_batch_list, y_batch_list, test_X_batch_list, test_y_batch_list = train_test_mini_batch(X_train, y_train, X_test, y_test, 4)

    # 모델 만들기(입력층, 은닉층, 출력층)
    model = Sequential([Input(8), Dense(8), Dense(1, activation='sigmoid')])

    # 모델 실행 및 정확도, 손실함수 측정
    print("<Train data>")
    # for i in range(10):
    #     evaluate_epoch(X_batch_list, y_batch_list, model, i)
    evaluate_epoch(X_batch_list, y_batch_list, model, 0)    # epoch 1
    print("<Test data>")
    evaluate_epoch(test_X_batch_list, test_y_batch_list, model, 0)

main()

# model = Sequential([Input(8), Dense(1, activation='sigmoid')])
# for i in range(10):
#     main()