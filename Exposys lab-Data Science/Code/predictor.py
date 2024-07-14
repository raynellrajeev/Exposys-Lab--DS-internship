import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

file = pd.read_csv('50_Startups.csv')

#replace missing values with mean value
mean = file.mean()
file.fillna(mean, inplace=True)

#data file
print(file.describe)

X = file.iloc[:, :-1].values #first 3 columns
y = file.iloc[:, -1].values #last column

# Plotting the line graph
plt.plot(file['R&D Spend'], label='R&D Spend')
plt.plot(file['Administration'], label='Administration Spend')
plt.plot(file['Marketing Spend'], label='Marketing Spend')
plt.plot(file['Profit'], label='Profit')
plt.xlabel('Index')
plt.ylabel('Spend')
plt.title('Spend for R&D, Administration and Marketing')
plt.legend()
plt.show()

rd_spend = float(input("Enter R&D Spend: "))
admin_cost = float(input("Enter Administration Cost: "))
marketing_spend = float(input("Enter Marketing Spend: "))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

def predict(regressor):

        regressor.fit(X_train, y_train)
        X_new = np.array([rd_spend, admin_cost, marketing_spend]).reshape(1,-1) #reshape 1d array to 2d array 
        prediction = regressor.predict(X_new)

        print("The predicted profit for the company is:", prediction[0])

        # Make predictions using the test data
        y_pred = regressor.predict(X_test)

        # Calculate the R-squared score(coefficient of determination)
        r2 = r2_score(y_test, y_pred) # compares y_test and y_pred

        # Print the R-squared score
        print("R2 score:", r2)
        
        #predicted values vs actual values
        results = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
        print(results)

        # Plotting the line graph
        plt.plot(y_test, label='Actual Values')
        plt.plot(y_pred, label='Predicted Values')

        plt.xlabel('Index')
        plt.ylabel('Values')

        plt.title('Actual vs Predicted Values')

        plt.legend()

        plt.show()