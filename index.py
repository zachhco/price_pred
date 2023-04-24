#Librarys
from flask import Flask, render_template, request
import math
from tkinter import X
import pandas as pd 
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

#read in the data
df = pd.read_csv('Book3.csv')

#Independent variables
X = df.drop('lnprice', axis = 1)

#seperate the predicting attriubut into Y for model training
y = df.lnprice

#split data
x_test, x_predict, y_test, y_predict = train_test_split(X, y)

# FITTING THE MODEL USING A PIPELINE
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_cols = ['hp2', 'speed', 'weight', 'year', 'acc', 'hp']
categorical_cols = ['seat', 'limit', 'top', 'rwd', 'na', 'tt', 'ttt', 'hyb']

mod = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[
                              ('model', mod)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(x_test, y_test)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(x_predict)

# Evaluate the model
score = mean_absolute_error(y_predict, preds)
print('MAE:', score)


###
####### FUNCTIONS ########
###

# Default function, takes the average values from the data and spits out a number!
def pr(hp=667, seat=2, top=.26, limit=.31, rwd=.63, year=2019, acc=3.0, na=.29, 
tt=.63, ttt=.03, hyb=.04, speed=202, weight=1581, hp2=477146):
    """ takes input values for all dependent vairbales and produces an output of a 
        cars predicted price. the default values are the mean values for all of the 
        cars in the dataset
        """

    # print(x_test)
    x_test.iat[0, 0] = hp
    x_test.iat[0, 1] = seat
    x_test.iat[0, 2] = top
    x_test.iat[0, 3] = limit
    x_test.iat[0, 4] = rwd
    x_test.iat[0, 5] = year
    x_test.iat[0, 6] = acc
    x_test.iat[0, 7] = na
    x_test.iat[0, 8] = tt
    x_test.iat[0, 9] = ttt
    x_test.iat[0, 10] = hyb
    x_test.iat[0, 11] = speed
    x_test.iat[0, 12] = weight
    x_test.iat[0, 13] = hp2
    A = x_test.to_numpy()
    # print(A)
    L = A.tolist()
    # print(L)
    N = x_test.columns.tolist()
    
    #print output
    print('_________________')
    print()
    for i in range(len(L[0])): 
        print (N[i], '  | ', L[0] [i])
    result = mod.predict(x_test)
    print('_________________')      #make pretty
    print()
    number = math.e**result[0]
    f = '{:,.2f}' .format(number)     #print with $ format
    print('This car is worth, $', f)
    # print('$', math.e**result[0])

def getinput(q, d):
    """takes q and default d
    """
    result = input(q)
    if len(result) < 1:
        return d
    else:
        return result

#APPLICATION    

app = Flask(__name__)

# Function to calculate predicted price
def prask(hp=667, seat=2, top=.26, limit=.31, rwd=.63, year=2019, acc=3.0, na=.29,
          tt=.63, ttt=.03, hyb=.04, speed=202, weight=1581, hp2=477146):
    """'prask' is the same as 'pr' except it asks for a user input on each explanatory variable.
        Output: predicted price of the car with the specifications that the user has put input"""

    # ask user for each input
    hp_input = request.form.get('hp', hp)
    seat_input = request.form.get('seat', seat)
    top_input = request.form.get('top', top)
    limit_input = request.form.get('limit', limit)
    rwd_input = request.form.get('rwd', rwd)
    year_input = request.form.get('year', year)
    acc_input = request.form.get('acc', acc)
    na_input = request.form.get('na', na)
    tt_input = request.form.get('tt', tt)
    ttt_input = request.form.get('ttt', ttt)
    hyb_input = request.form.get('hyb', hyb)
    speed_input = request.form.get('speed', speed)
    weight_input = request.form.get('weight', weight)
    
    # Convert input values to appropriate data types
    hp = int(hp_input)
    seat = int(seat_input)
    top = float(top_input)
    limit = float(limit_input)
    rwd = float(rwd_input)
    year = int(year_input)
    acc = float(acc_input)
    na = float(na_input)
    tt = float(tt_input)
    ttt = float(ttt_input)
    hyb = float(hyb_input)
    speed = float(speed_input)
    weight = float(weight_input)
    hp2 = float(hp)**2

    x_test.iat[0, 0] = hp
    x_test.iat[0, 1] = seat
    x_test.iat[0, 2] = top
    x_test.iat[0, 3] = limit
    x_test.iat[0, 4] = rwd
    x_test.iat[0, 5] = year
    x_test.iat[0, 6] = acc
    x_test.iat[0, 7] = na
    x_test.iat[0, 8] = tt
    x_test.iat[0, 9] = ttt
    x_test.iat[0, 10] = hyb
    x_test.iat[0, 11] = speed
    x_test.iat[0, 12] = weight
    x_test.iat[0, 13] = hp2
    A = x_test.to_numpy()
    L = A.tolist()
    N = x_test.columns.tolist()

    # calculate predicted price
    result = mod.predict(x_test)
    number = math.e**result[0]
    f = '{:,.2f}'.format(number)

    # Return the predicted price as a response
    #return f"This car is worth: ${f}"
    return render_template('results.html', f=f)

@app.route('/')
def mod_dashboard():
    return render_template('calculate.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    # Call the prask() function to calculate the predicted price
    result = prask()

    # Return the result
    return result

if __name__ == '__main__':
    app.run()
