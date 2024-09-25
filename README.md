# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare your data
2. Define your model
3. Define your cost function
4. Define your learning rate
5. Train your model
6. Evaluate your model
7. Tune hyperparameters
8. Deploy your model

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Shiv Sujan S R
RegisterNumber: 212223040194
*/
```
```py
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
### Initial data set:
<img width="507" alt="6 1" src="https://github.com/Rajeshanbu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118924713/bece671e-ccd0-4675-ae64-8179316b89b3">

### Data info:
<img width="157" alt="6 2" src="https://github.com/Rajeshanbu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118924713/53102c1f-aab6-4f24-a2e0-7692f8e06f18">

### Optimization of null values:
<img width="143" alt="6 3" src="https://github.com/Rajeshanbu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118924713/80476a85-9c0f-4de1-8329-fb23ec72e5d2">

### Assignment of x and y values:
<img width="173" alt="6 4" src="https://github.com/Rajeshanbu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118924713/9c634ef3-962d-4480-ad02-364970fe4331">

<img width="677" alt="6 5" src="https://github.com/Rajeshanbu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118924713/194f2f76-4a18-4f95-bee2-16fa348d3a43">

### Converting string literals to numerical values using label encoder:
<img width="594" alt="6 6" src="https://github.com/Rajeshanbu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118924713/b64703a3-3985-41d7-a6d9-f68b580faff7">

### Accuracy:
<img width="124" alt="6 7" src="https://github.com/Rajeshanbu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118924713/5aad200e-49b6-4e23-84a2-f627fa4f6c50">

### Prediction:
<img width="673" alt="6 8" src="https://github.com/Rajeshanbu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118924713/b04c8de7-696c-464c-851f-f39ec95511dd">

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

