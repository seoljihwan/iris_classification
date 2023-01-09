from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

Iris= load_iris()

Iris_Data = pd.DataFrame(data=np.c_[Iris['data'],Iris['target']],columns=Iris["feature_names"]+['target'])
Iris_Data['target'] = Iris_Data['target'].map({0:"steosa",1:"versicolor",2:"virginica"})

Iris_Data.to_csv("iris.csv")