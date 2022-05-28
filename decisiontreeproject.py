from pandas.core.common import random_state
from google.colab import files
data_to_upload = files.upload()
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from io import StringIO
from IPython.display import Image
import pydotplus
from sklearn.tree import export_graphviz

col_names = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'label']
df = pd.read_csv("titanic.csv", names = col_names).iloc[1:]
features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = df[features]
y = df.label
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 1)
dv = DecisionTreeClassifier()
dv = dv.fit(train_X, train_y) 
y_prediction = dv.predict(test_X)
print("the accuracy of your pathetic model is" , accuracy_score(test_y, y_prediction))
dot_data = StringIO()
export_graphviz(dv, out_file = dot_data, filled = True , rounded = True, special_characters = True, feature_names = features, class_names = ["0", "1"])
print(dot_data.getvalue())
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("uwu.png")
Image(graph.create_png())
dv = DecisionTreeClassifier(max_depth = 3)
dv = dv.fit(train_X, train_y)
y_prediction = dv.predict(test_X)
print("the accuracy of your pathetic model is" , accuracy_score(test_y, y_prediction))
dot_data = StringIO()
export_graphviz(dv, out_file = dot_data, filled = True , rounded = True, special_characters = True, feature_names = features, class_names = ["0", "1"])
print(dot_data.getvalue())
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("uwu.png")
Image(graph.create_png())