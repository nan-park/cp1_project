import numpy as np

# Xavier 초기화(활성화함수가 시그모이드일 때 잘 동작). 편향 포함한 후에 분리하기
# 이전 층 노드 개수가 n, 현재 층 노드 개수 m일 때, 표준편차가 2/루트(n+m)인 정규분포로 초기화
def initialize_parameter(n, m):
  init = np.random.normal(0, 2/((n+m)**2), (n+1, m))
  W = init[:-1, :]
  b = init[-1, :]
  return W, b

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Layer():  # 입력층, 은닉층, 출력층
    def __init__(self, node_num, activation='linear'):
        self.node_num = node_num    # 레이어의 노드 개수
        self.activation = activation    # 활성화 함수
        self.prev = None    # 이전 층
        self.next = None    # 다음 층

    def set_weights(self):  # 가중치 행렬, 편향 초기화
        if self.prev is not None:
            prev_node_num = self.prev.node_num
            self.W, self.b = initialize_parameter(prev_node_num, self.node_num)

    # input 행렬 X
    @property
    def X(self):
        return self._X
    @X.setter
    def X(self, value):
        self._X = value

class Dense(Layer): # 은닉층, 출력층
    def __init__(self, node_num, activation='linear'):
        # super().__init__(self, node_num)
        self.node_num = node_num
        self.activation = activation
        self.prev = None
        self.next = None

    # input을 받아 output 출력
    def output(self):
        answer = np.dot(self._X, self.W) + self.b
        if self.activation == 'linear': # 활성화함수 없음(선형함수)
            return answer
        elif self.activation == 'sigmoid':  # 활성화함수: 시그모이드
            answer = sigmoid(answer)
            return answer

class Input(Layer): # 입력층
    def __init__(self, node_num, activation='linear'):
        # super().__init__(self, node_num)
        self.node_num = node_num
        self.activation = activation
        self.prev = None
        self.next = None

    # 입력층의 경우 input을 그대로 출력한다
    def output(self):
        return self._X



# Sequential([])에 Layer 쌓고 서로 연결되도록 하기. 가중치 초기화 가능해야 함
class Sequential():
    def __init__(self, layer_list): # Layer들을 서로 링크드리스트로 연결. 처음과 끝 지정.
        # layer가 없는 경우
        if len(layer_list)==0:
            self.head = None
            self.tail = None
        # layer가 1개인 경우
        elif len(layer_list)==1:
            self.head = layer_list[0]
            self.tail = layer_list[0]
        else:   # layer가 2개 이상인 경우
            self.head = layer_list[0]
            iterator = self.head
            for layer in layer_list[1:-1]:
                layer.prev= iterator
                iterator.next = layer
                iterator = layer
            iterator.next = layer_list[-1]
            self.tail = layer_list[-1]
            self.tail.prev = iterator

        # 가중치, 편향 초기화
        iterator = self.head
        while iterator:
            iterator.set_weights()
            iterator = iterator.next

    # input으로 들어올 X 행렬
    @property
    def input(self):
        return self._input
    @input.setter
    def input(self, value):
        self._input = value
        self.head.X = value

    # Layer를 모두 거쳐 output을 출력
    def output(self):
        iterator = self.head  # Input
        while iterator.next:
            iterator.next.X = iterator.output()
            iterator = iterator.next
        return iterator.output()