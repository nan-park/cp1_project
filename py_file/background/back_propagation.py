import numpy as np

def back_prop(model, y_pred_prob, y_batch, learning_rate=1):
    def sigmoid_prime(y):
        return y * (1 - y)
    # input(X), output(predict), y, error(cross_entropy)
    def cross_entropy_prime(y_pred_prob, y_batch):  # binary cross entropy
        return y_pred_prob - y_batch

    iterator = model.tail
    while iterator != model.head:
        # dError/dY
        if iterator == model.tail:
            error = cross_entropy_prime(y_pred_prob, y_batch)
        else:
            error = np.dot(delta, iterator.next.W.T)    # (체크)
        # dY/dy
        if iterator.activation == 'sigmoid':
            delta = error * sigmoid_prime(y_pred_prob)
        elif iterator.activation == 'linear':
            delta = error
        # dy/dw
        delta_mean = np.mean(delta)
        iterator.b -= learning_rate * delta_mean
        iterator.W -= learning_rate * np.dot(iterator.X.T, delta)
        iterator = iterator.prev

