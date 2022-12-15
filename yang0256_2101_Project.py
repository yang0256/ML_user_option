"""
Created on Fri Dec  2 22:54:07 2022

@author: Tao Yang , project for CST2101 Business Intelligence Programming
The purpose of this program is to demonstrates the correct use of functions and/or 
classes and methods, along with approprate user interaction for each step with the 
ability to exit or restart at any step.  


Submittance: 
yang0256-2101-Project.zip
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
import numpy as np


def err_mesg(err_count):
    print("Not an appropriate choice.")
    err_count += 1  
    # print(err_count)    

    if err_count > 3:
        print("Sorry, too many errors. Please restart. \n\n")
        
        sys.exit()
    else:
        return err_count
        
def ask_user_continue():
    '''define function for screen break '''
    while True:    
        user_next = input("Continue: y/n? ")  
        if user_next == 'y':
            return
        elif user_next == 'n':
            sys.exit()
        else:
            print('Please enter y/n ....\n')         

def let_user_choose(options):   
     
    '''define funtion for user to enter choice and validate the input'''
    
    print("Please choose dataset:")
    # list the dataset choices1
    
    for id, choice in enumerate(options):    
        print("{}) {}".format(id + 1, choice))  

    # Validation handling
    err_count = 0
    
    while True:
        # i the choice number
        idx = input("Enter choice number: ")    

        # Validation Rules
        try:
            ### if numberic value entered
            id = int(idx) -1
            
            if id in range(len(options)):
                
                return id
            
            else:
              
                err_count = err_mesg(err_count)

        except ValueError:
            ### if other input error
            err_count = err_mesg(err_count)

def load_dataset_cali():
    '''define function to load the california housing dataset'''
    
    from sklearn.datasets import fetch_california_housing
    california = fetch_california_housing()
    # california_housing.frame.info()
    pd.set_option('display.precision', 4)
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.width' , 400)
    california_df = pd.DataFrame(california.data, columns = california.feature_names)
    california_df['MedHouseValue'] = pd.Series(california.target)
    
    # print(california_df.head())
    # print(california_df.describe())
    
    sample_df = california_df.sample(frac=0.1, random_state=17)
    print(sample_df.head())
    # print(sample_df.describe())
    print('\n\nCalifornia_housing datdaset loaded: \n\n')
    
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    sns.set(font_scale=2)
    sns.set_style('whitegrid')
    
    ask_user_continue()
    
    # '''plotting the eight features as X and MedHouseValue as Y'''
    # for feature in california.feature_names:
    #     plt.figure(figsize=(16, 9))
    #     sns.scatterplot(data=california_df, x=feature,
    #         y='MedHouseValue', hue='MedHouseValue',
    #         palette='cool', legend=False)
    #     # print('8 features')
        
    #     plt.show()
    #     print('Chart feature is', feature)
    #     ask_user_continue()

    return california


def split_train_test(sample_df):
    ''' function to split the sample data into model's train and test data group'''    
    ###Splitting the Data for Training and Testing
    # from sklearn.datasets import fetch_california_housing
    # california = fetch_california_housing()
    from sklearn.model_selection import train_test_split
    
    ## randomize and split the arrays of sample and target data, seed value returns
    ## 4 tuples, X is training 2D array, y is target portion 1D array
    X = sample_df.data
    y = sample_df.target
    
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, random_state=11) 

    # print("X_train array", X_train.shape)

    # print("X_test", X_test.shape)

    return  X_train, X_test, y_train, y_test 


def LinearRegression_training(X_train, X_test, y_train, y_test, sample_df):
    ''' create a LinearRegression estimator and invoke its fit method to train the estimator using X_train and y_train'''
    from sklearn.linear_model import LinearRegression
    linear_regression = LinearRegression()
    linear_regression.fit(X=X_train, y=y_train)
    for i, name in enumerate(sample_df.feature_names):
        print(f'{name:>10}: {linear_regression.coef_[i]}')

    predicted = linear_regression.predict(X_test)
    expected = y_test

    # print('\n\npredited 5: ',predicted[:5])
    # print('expected 5: ',expected[:5])
    
    return expected,predicted


def visualize_regression(expected,predicted):
    df = pd.DataFrame()
    df['Expected'] = pd.Series(expected)
    df['Predicted'] =pd.Series(predicted)
    # figure = plt.figure(figsize=(9, 9))
    axes = sns.scatterplot(data=df, x='Expected', y='Predicted', hue='Predicted', palette='cool', legend=False)
    start = min(expected.min(), predicted.min())
    end = max(expected.max(), predicted.max())
    axes.set_xlim(start, end)
    axes.set_ylim(start, end)
    
    
    plt.plot([start, end], [start, end], 'k--')
    plt.show()

def load_dataset_diabetes():
    '''define function to load the diabetes dataset'''    
    # import matplotlib.pyplot as plt
    # import numpy as np
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    
    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]
    
    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    
    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)
    
    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)
    
    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))
    
    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
    plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    
    plt.show()
    
    ask_user_continue()


def user_interaction_main():
    '''define main menu user interaction function '''    
    while True:
        try:
            # call func for input validation and error handling, parsing choice#
            user_choice = let_user_choose(options)   
        except ValueError:
            #ask user if to continute at errors
            ask_user_continue()            
            continue
            
        if (user_choice == 0):                
            sample_df = load_dataset_cali()
            # print(sample_df)      

            # Split the data into training and testing sets

            X_train, X_test, y_train, y_test = split_train_test(sample_df)
 
            print("\n\nSplit the date into training and test sets \n\nX_train array", X_train.shape)
            print("X_test array", X_test.shape)
 
            ask_user_continue()   
 
            print("\n\nCoefficients for each feature:\n\n")
            expected,predicted = LinearRegression_training(X_train, X_test, y_train, y_test, sample_df)
            
            
            # # Fit a linear regression model to the training data
            # model = fit_linear_model(X_train, y_train)

            # # Evaluate the model on the testing data
            # evaluate_model(model, X_test, y_test)
            visualize_regression(expected,predicted)
            
            ask_user_continue()  
                
        elif (user_choice == 1):
            load_dataset_diabetes()
            # ask_user_continue()
            
            
        elif (user_choice == 2):
            print ("Exiting the program")
       
            sys.exit()
            
        else:
            
            print('Please choose from the list.\n\n\n')
            continue
        
        
        
'''enter please choose dataset 1 2  and exit 3 '''
options = ["California Housing Dataset",  
           "Diabetes",  
           "Exit"]



user_interaction_main()

