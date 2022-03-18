import pandas as pd
import numpy as np

from sklearn.utils import shuffle

class Rating_Matrix():
    
    def __init__(self, file_path, train_size = 0.8, missing_value=0.0):
        """
        MF 모델 사용을 위한 MovieLens 1M 데이터셋으로 rating matrix을 생성한다.

        Args:
            file_path (string): MovieLens 1M 데이터셋의 위치
            train_size (float, optional): 분할 할 training set의 크기 비율. Defaults to 0.8.
            missing_value (float, optional) : 결측치의 기본값을 0.0으로 설정. Default to 0.
        """
        
        self.file_path = file_path
        self.train_size = train_size
        self.missing_value = missing_value
        
    def preprocessing(self):
        """
        등급 행렬을 만든다.
        
        return: [train dataset, test dataset]
        """
        r_cols = ["UserIDs", "MovieIDs", "Ratings", "Timestamp"]
        ratings = pd.read_csv(self.file_path, names=r_cols, sep="::", engine="python", encoding="utf-8")
        ratings = ratings[["UserIDs", "MovieIDs", "Ratings"]].astype(int) # Timestamp 제거
        
        rating_matrix = ratings.pivot(index="UserIDs", columns="MovieIDs", values="Ratings").fillna(self.missing_value)
        return rating_matrix