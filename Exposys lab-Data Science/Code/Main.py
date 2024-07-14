import predictor

while(True):
    choice= int(input("\n1.Linear Regression\n2.Decision Tree Regression\n3.Random Forest Regression\n4.Support Vector Regression\n5.EXIT\n"))
    if (choice==1):

        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        predictor.predict(regressor)

    elif (choice==2):

        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor()
        predictor.predict(regressor)

    elif (choice==3):

        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor()
        predictor.predict(regressor)

    elif (choice==4):

        from sklearn.svm import SVR
        regressor = SVR(kernel='linear')
        predictor.predict(regressor)

    elif (choice==5):

        print("\nThank you for using the program")
        break
    
    else:
        print("invalid choice")