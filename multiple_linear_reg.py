import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer 
import statsmodels.formula.api as sm
import statsmodels.api as sm
import statsmodels.regression.linear_model as sm


class multiple_linear_regression:
    def __init__(self, file):
        self.file = file
    def data_selection(self, a, b,c):
       global rx, ry
       data = pd.read_csv(self.file)
       rx = data.iloc[:,a:b].values
       ry = data.iloc[:,c].values
    
    def filter_dataset(self):
        global rx, ry
        # for encoding  purpose, use the label that is in a categorical form, the third column is categorical (0,1,2,3)
        prompt = int(input("Enter the number of columns that are in a categorical form: "))
        le = LabelEncoder()
        if prompt >= 1:
            for i in range(1,prompt+1):
                column = int(input("Enter the column number: "))
                rx[:,column] = le.fit_transform(rx[:,column])
                ct = ColumnTransformer([("State", OneHotEncoder(), [column])], remainder = 'passthrough')
                rx = ct.fit_transform(rx)
        elif prompt == 0:
            oneh = OneHotEncoder(categories = "auto")
            rx = oneh.fit_transform(rx).toarray()
        else:
            raise ValueError('value must be equal or greater than zero')
    
    def trainig(self, ts):
        global linear_reg, training_x, training_y, testing_x,testing_y
        training_x,testing_x,training_y,testing_y = train_test_split(rx,ry,test_size = ts, random_state = 0)
        linear_reg = LinearRegression()
        # start trainig with fit method
        linear_reg .fit(training_x,training_y)
    
    def predict(self, k):
       global pred_y
       pred_y =linear_reg.predict(testing_x)
       print(testing_y[k])
       print(pred_y[k])
    
    def accuracy(self):
        print(linear_reg.score(testing_x, testing_y))
        
        

    def backward_elimination(self):
        global rx
        rx = np.append(arr= np.ones((50,1)).astype(int), values = rx , axis = 1)
        # ordinary least square 
        x_opt = rx[:, [2,3,5]]
        reg_ols = sm.OLS(endog = ry, exog= x_opt).fit()
        print(reg_ols.summary())


# do not forget r
multiple_linear_regression = multiple_linear_regression("multiple_linear_reg.csv")
multiple_linear_regression.data_selection(0,4, 4)
multiple_linear_regression.filter_dataset()
multiple_linear_regression.trainig(0.2)
multiple_linear_regression.predict(3)
multiple_linear_regression.accuracy()

#  1, 3