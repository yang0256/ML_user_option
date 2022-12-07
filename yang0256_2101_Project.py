"""
Created on Fri Dec  2 22:54:07 2022

@author: Tao Yang , project for CST2101 Business Intelligence Programming
The purpose of this program is to demonstrates the correct use of functions and/or 
classes and methods, along with approprate user interaction for each step with the 
ability to exit or restart at any step.  


Submittance: yang0256-2101-Project.zip
The zip file includes 
1) main program: yang0256-2101-Project.py 
3) design document: yang0256-Design.docx
4) test documents: test plan functionality and exception handling



"""
''' required classes and modules'''
import sys 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''define funtion for user to enter choice and validate the input'''
def let_user_choose(options):   
    print("Please choose dataset:")
    # list the dataset choices
    for id, choice in enumerate(options):    
        print("{}) {}".format(id + 1, choice))
    

    # Validation handling
    err_count = 0
    
    while True:
        # i the choice number
        i = input("Enter number: ")    
        
        # Validation Rules
        if i not in ('1', '2', '3', '4'):
            print("Not an appropriate choice.")
            err_count+=1
            
            if err_count > 4 :
                print("Sorry, too many errors")
                break
        else:
            #choice parsed and exit the while loop
            break    
    #parsing the choice value
    try:
        if 0 < int(i) <= len(options):           
            
            return int(i) - 1              

    except TypeError:
        print("Exception Error Here")        

'''define function for screen break '''
def ask_user_continue(): 
    
    while True:
        
        user_next = input("Continue: y/n? ")  
        
        if user_next == 'y':
            return
        elif user_next == 'n':
            sys.exit()
        else:
            print('Please enter y/n ....')
            
    
'''define function to load the california housing dataset'''
def load_dataset_cali():
    from sklearn.datasets import fetch_california_housing
    california = fetch_california_housing()
    pd.set_option('display.precision', 4)
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.width' , 300)
    california_df = pd.DataFrame(california.data, columns = california.feature_names)
    california_df['MedHouseValue'] = pd.Series(california.target)
    
    # print(california_df.head())
    # print(california_df.describe())
    
    sample_df = california_df.sample(frac=0.1, random_state=17)
    # print(sample_df)
    
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    sns.set(font_scale=2)
    sns.set_style('whitegrid')
    
    
    # let_user_choose_next()

    print(sample_df.head())
    print(sample_df.tail())
    print('......California_housing datdaset loaded:')
    
    ask_user_continue()
    
    # print('Plot chart 1')
    ## plotting the eight features as X and MedHouseValue as Y
    for feature in california.feature_names:
        plt.figure(figsize=(16, 9))
        sns.scatterplot(data=sample_df, x=feature,
            y='MedHouseValue', hue='MedHouseValue',
            palette='cool', legend=False)
        print('8 features')



    # print('Plot chart 2')1
# sample_df =[]
''' function to split the sample data into model's train and test data group'''
def split_train_test(sample_df):
    
    ###Splitting the Data for Training and Testing

    from sklearn.model_selection import train_test_split
    
    
    X_train, X_test, y_train, y_test = train_test_split(
         sample_df.data, sample_df.target, random_state=11)

    # print('Plot chart 3')
    X_train.shape
    (15480, 8)
    # print('Plot chart 4')
    X_test.shape
    (5160, 8)
    # print('Plot chart 5')


def user_interaction_main():
    
    while True:
        try:
            # call func for input,validation and error handling, parsing choice value
            user_choice = let_user_choose(options)   
            
        except ValueError:
            #ask user if to continute at errors
            
            ask_user_continue()
            
            continue
            
        if (user_choice == 0):
            
            
            sample_df = load_dataset_cali()
                
            ask_user_continue()
            
            # split_train_test(sample_df)1
            
                
        elif (user_choice == 1):
            load_dataset_cali()  
            
        elif (user_choice == 2):   
            load_dataset_cali() 
            
        elif (user_choice == 3):
            print ("Exiting the program")
       
            sys.exit()
        else:
            break

options = ["California Housing Dataset",  
           "Load 2",  
           "Load 3", 
           "Exit"]



user_interaction_main()

# def user_interaction():

#     if (user_choice == 0):
#             load_dataset_cali()
#     elif (user_choice == 1):
#         load_dataset_cali()  
#     elif (user_choice == 2):   
#         load_dataset_cali() 
#     elif (user_choice == 3):
#         print ("Exiting the program")
#         sys.exit()
#     else:
#         print ("Exit")
    ##exit
    ##
    

# print(california.DESCR)

# print(california.data.shape)
# print(california.target)
# print(california.feature_names)

'''
# import pandas as pd
pd.set_option('display.precision', 4)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)q

california_df = pd.DataFrame(california.data, columns = california.feature_names)
california_df['MedHouseValue'] = pd.Series(california.target)

# print(california_df.head())
# print(california_df.describe())

# sample_df = california_df.sample(frac=0.1, random_state=17)
# print(sample_df)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
sns.set_style('whitegrid')


for feature in california.feature_names:
    plt.figure(figsize=(16, 9))
    sns.scatterplot(data=sample_df, x=feature,
        y='MedHouseValue', hue='MedHouseValue',
        palette='cool', legend=False)

###Splitting the Data for Training and Testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     california.data, california.target, random_state=11)

X_train.shape
(15480, 8)
X_test.shape
(5160, 8)

##Training the Model

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X=X_train, y=y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
         normalize=False)

for i, name in enumerate(california.feature_names):
      print(f'{name:>10}: {linear_regression.coef_[i]}')

###Testing the Model

predicted = linear_regression.predict(X_test)
expected = y_test
###first five predictions and their corresponding expected values:

predicted[:5]
# Out[32]: array([1.25396876, 2.34693107, 2.03794745, 1.8701254, 2.53608339])

expected[:5]
# Out[33]: array([0.762, 1.732, 1.125, 1.37 , 1.856])

df = pd.DataFrame()
df['Expected'] = pd.Series(expected)
df['Predicted'] = pd.Series(predicted)

##plot the data as a scatter plot with the expected (target) prices 
##along the x-axis and the predicted prices along the y-axis
figure = plt.figure(figsize=(9, 9))
axes = sns.scatterplot(data=df, x='Expected', y='Predicted',
         hue='Predicted', palette='cool', legend=False)
##set the x- and y-axes’ limits
start = min(expected.min(), predicted.min())
end = max(expected.max(), predicted.max())
axes.set_xlim(start, end)
axes.set_ylim(start, end)
##plot a line that represents perfect predictions 
line = plt.plot([start, end], [start, end], 'k--')


### Regression Model Metrics

from sklearn import metrics
metrics.r2_score(expected, predicted)
##call function mean_squared_error (from module sklearn.metrics) 
metrics.mean_squared_error(expected, predicted)





###############################################################
'''
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:39:04 2022

@author: solvm
"""

'''

1)user interaction:choose from three datasets, give user 3 options to choose from
no local static path required
2)head tail

3
error handler
try:
    except error:
        
        handle

4 training data model
5 variable myInteger or my_integar  <<< for variables use meaningful names
MyClass <<< for classes


6 use //// doc string, proper header, inline doc

7 program design logic  

test plan, copy paste test and exception handling <<<

design and test documents 


yang0256-Program.py
yang0256-TestPlan.xls
yang0256-2101-ProjectDATA.zip
'''

# print(sys.prefix)