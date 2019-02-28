# coding=utf-8
# Linear Regression Realization (http://dataaspirant.com/2014/12/20/linear-regression-implementation-in-python/)

# input_data.csv:
# square_feet;price
# 150;6450
# 200;7450
# 250;8450
# 300;9450
# 350;11450
# 400;15450
# 600;18450

# Required Packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model



# Function to get data
def get_data(file_name):
    global x_parameter, y_parameter
    try:
        # TODO: Replace the separator (sep) with your value.
        data = pd.read_csv(file_name, sep=";")
        x_parameter = []
        y_parameter = []
        # TODO: Replace the names of the fields 'square foot', 'price' for your own values
        for single_square_feet in data['square_feet']:
            x_parameter.append([float(single_square_feet)])

        for single_price_value in data['price']:
            y_parameter.append(float(single_price_value))
    except IOError:
        exit('File {0} not found. Exit'.format(file_name))

    return x_parameter, y_parameter


# Function for Fitting our data to Linear model
# noinspection PyPep8Naming
def linear_model_main(x_parameters, y_parameters, predict_value):
    lin_reg_score = []
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(x_parameters, y_parameters)
    # noinspection PyArgumentList
    predict_outcome = regr.predict(predict_value)
    r2score = regr.score(x_parameters, y_parameters)
    lin_reg_score.append(r2score)
    predictions = {'intercept': regr.intercept_, 'coefficient': regr.coef_, 'predicted_value': predict_outcome,
                   'r2score': r2score}
    return predictions


# Function to show the resutls of linear fit model
def show_linear_line(x_parameters, y_parameters):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(x_parameters, y_parameters)
    plt.scatter(x_parameters, y_parameters, color='blue')
    # noinspection PyArgumentList
    plt.plot(x_parameters, regr.predict(x_parameters), color='red', linewidth=4)
    # Supress axis value
    plt.xticks(())
    plt.yticks(())
    plt.show()


X, Y = get_data('input_data.csv')
predicted_value = np.array([700])  # 700
predicted_value = predicted_value.reshape(1, -1)
result = linear_model_main(X, Y, predicted_value)
constant = result['intercept']
coefficient = result['coefficient'][0]
predicted_value = result['predicted_value'][0]
r_square = result['r2score']
print('Constant Value: {0}'.format(round(constant, 2)))
print('Coefficient: {0}'.format(round(coefficient, 2)))
print('Predicted Value: {0}'.format(round(predicted_value, 2)))
print('R-Square: {0} ({1}%)'.format(r_square, round(r_square, 4) * 100))
show_linear_line(X, Y)
