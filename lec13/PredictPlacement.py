import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "placements.csv")
ds = pd.read_csv(csv_path)
# print(ds)

# steps:
    # 0. Preprocess + EDA + Feture selection
    #   1. Extract input and output cols
    #   2. Scale the values
    #  3.Train test split
    # 4. Train the model
    # 5. Evaluation and model selection
    # 6. Deploy the model
    
# print(ds.info())

# preprocessing

ds = ds.iloc[:, 1:]
print(ds.info())

# Eda
plt.scatter(ds['cgpa'], ds['iq'], c=ds['placement'])
# plt.show()


# algo - logistc regression can be use in this 

x = ds.iloc[:, 0:2] # independent variable(Input)
y = ds.iloc[: , -1] # dependent variable(Output)


# train test split
from sklearn.model_selection import train_test_split
x_train,  x_test , y_train , y_test = train_test_split(x,y,test_size=0.1)
# print(y_test)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) #scale from -1 to +1
# print(x_train)

x_test = scaler.transform(x_test)
# print(x_test)


# train model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(x_train, y_train)


ypred = clf.predict(x_test)
print(ypred)
print(y_test)


# find accuracy of model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, ypred))


# know the partition line 
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
plot_decision_regions(x_train, y_train.values , clf=clf, legend=2)
# plt.show()



# to convert the model into a file
import pickle
pickle.dump(clf,open("model.pkl","wb"))