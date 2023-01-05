import numpy as np

def predict(X_batch, y_batch, model):
    model.input = X_batch
    y_pred_prob = model.output()
    return y_pred_prob


def evaluate(X_batch_list, y_batch_list, model):
    def accuracy(y_pred, y_batch):
        return sum(y_pred == y_batch)[0] / y_pred.shape[0]

    def cross_entropy(y_pred_prob, y_batch):
        delta = 1e-7
        return -np.sum(y_batch * np.log(y_pred_prob + delta)) / y_batch.shape[0]  # (체크)

    def classification(x):
        if x < 0.5:
            return 0
        else:
            return 1

    accuracy_list = []
    cross_entropy_list = []
    classify = np.vectorize(classification)
    for i in range(len(X_batch_list)):  # (체크) evaluate 기능만 남겨두고 다시 메서드화하기
        X_batch = X_batch_list[i]
        y_batch = y_batch_list[i]
        y_pred_prob = predict(X_batch, y_batch, model)
        y_pred = classify(y_pred_prob)

        accuracy_list.append(accuracy(y_pred, y_batch))
        cross_entropy_list.append(cross_entropy(y_pred_prob, y_batch))

    accuracy = np.mean(accuracy_list)
    cross_entropy = np.mean(cross_entropy_list)
    return accuracy, cross_entropy
