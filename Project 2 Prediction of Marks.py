#Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Step 2: Import datasets
url="http://bit.ly/w-data"
data=pd.read_csv(url)
print("Data imported successfully")
data.head(10)

# Plotting the distribution of score
data.plot(x='Hours', y='Scores', style='-')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

#Step 3: Splitting data into training and test sets
x = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values 

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0) 

#Step 4: Training model on training set
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(x_train, y_train) 

print("Training complete.")

# Plotting the regression line
line = regressor.coef_*x+regressor.intercept_

# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line)
plt.title("Comparison")
plt.show()

print(x_test) # Testing data - In Hours
y_pred = regressor.predict(x_test) # Predicting the scores

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df) 

score_pred=np.array([7])
score_pred=score_pred.reshape(-1,1)
predict=regressor.predict(score_pred)
print("No of hours={}".format(7))
print("Predicted Score={}".format(predict[0]))

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 