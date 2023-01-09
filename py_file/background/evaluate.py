import numpy as np
from background.back_propagation import back_prop

def predict(X_batch, y_batch, model):
    model.input = X_batch
    y_pred_prob = model.output()
    return y_pred_prob


def evaluate_mini_batch(X_batch, y_batch, model, i):
    def accuracy(y_pred, y_batch):  # 정확도
        return sum(y_pred == y_batch)[0] / y_pred.shape[0]

    def cross_entropy(y_pred_prob, y_batch):    # 손실함수
        delta = 1e-7
        return -np.mean(y_batch * np.log(y_pred_prob + delta) + (1 - y_batch) * np.log(1 - y_pred_prob + delta))   # (체크)

    def classification(x):
        if x < 0.5:
            return 0
        else:
            return 1
        # return 0 if x < 0.5 else 1
        # return round(x, 0)

    classify = np.vectorize(classification)
    y_pred_prob = predict(X_batch, y_batch, model)
    y_pred = classify(y_pred_prob)
    accuracy = accuracy(y_pred, y_batch)
    cross_entropy = cross_entropy(y_pred_prob, y_batch)
    print(f"[Mini-Batch {i+1}] Loss = {round(cross_entropy, 3)}, Accuracy = {round(accuracy, 3)}")
    back_prop(model, y_pred_prob, y_batch)
    # print(model.tail.W)
    return accuracy, cross_entropy

def evaluate_epoch(X_batch_list, y_batch_list, model, epoch):
    accuracy_list = []
    cross_entropy_list = []
    for i in range(len(X_batch_list)):
        X_batch = X_batch_list[i]
        y_batch = y_batch_list[i]
        accuracy, cross_entropy = evaluate_mini_batch(X_batch, y_batch, model, i)
        accuracy_list.append(accuracy)
        cross_entropy_list.append(cross_entropy)

    total_accuracy = np.mean(accuracy_list)
    total_cross_entropy = np.mean(cross_entropy_list)
    print(f"[Epoch {epoch+1}] Loss = {round(total_cross_entropy, 3)}, Accuracy = {round(total_accuracy, 3)}\n")