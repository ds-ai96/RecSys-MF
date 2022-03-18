import numpy as np

class MF():
    
    def __init__(self, R, K, alpha, beta, iterations):
        """
        등급 행렬의 빈 부분을 예측하기 위해서 MF을 수행함

        Args:
            R (DataFrame): 사용자-아이템 등급 행렬
            K (int): 잠재 공간의 차원 수
            alpha (float): 학습률 (gamma : in paper)
            beta (float): 규제화 파라미터 (lambda : in paper)
            iterations (int): 반복 횟수
        """
        
        self.R = np.array(R)
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self. iterations = iterations
        
    def train(self):
        """
        모델의 훈련을 진행하고 훈련 과정을 반환함.
        
        Returns:
            list: 훈련 반복 횟수 마다의 error(rmse) 값
        """
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))
        
        # Initialize the biases term
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.mu = np.mean(self.R[self.R.nonzero()])
        
        # Create a list of samples
        rows, columns = self.R.nonzero()
        self.samples = [
            (u, i, self.R[u, i])
            for u, i in zip(rows, columns)
        ]
        
        # 실제 train을 진행하는 부분 (SGD를 사용)
        training_process = []
        for epoch in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse = self.rmse()
            training_process.append((epoch+1, rmse))
            if (epoch + 1) % 1 == 0:
                print(f"Iteration: {epoch+1} ; Error = {rmse:.4f}")
        return training_process
        
    def rmse(self):
        """
        Root Mean Square Error을 계산하기 위한 함수
        """
        xs, ys = self.R.nonzero()
        pred_ratings = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - pred_ratings[x, y], 2)
        return np.sqrt(error / xs.shape[0])
    
    def sgd(self):
        for u, i, r in self.samples:
            prediction = self.get_rating(u, i)
            e = (r - prediction)
            
            # update biases
            self.b_u[u] += self.alpha * (e - self.beta * self.b_u[u])
            self.b_i[i] += self.alpha * (e - self.beta * self.b_i[i])
            
            # updata latent factors
            self.P[u, :] += self.alpha * (e * self.Q[i, :] - self.beta * self.P[u, :])
            self.Q[i, :] += self.alpha * (e * self.P[u, :] - self.beta * self.Q[i, :])
            
    def get_rating(self, u, i):
        """
        사용자 u와 아이템 i의 예측 등급을 반환하는 함수
        """
        pred_rating = self.mu + self.b_u[u] + self.b_i[i] + self.P[u, :].dot(self.Q[i, :].T)
        return pred_rating
    
    def full_matrix(self):
        """
        결측치를 전부 계산한 full matrix을 반환하는 함수
        """
        return self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis,] + self.P.dot(self.Q.T)