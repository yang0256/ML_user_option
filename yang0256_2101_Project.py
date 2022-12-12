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



# def err_mesg(err_count):
#     err_count +=1          
#     if err_count > 3:
#         print("Sorry, too many errors")
#         break
#     else:
#         return


def let_user_choose(options):   
     
    '''define funtion for user to enter choice and validate the input'''
    
    print("Please choose dataset:")
    # list the dataset choices
    for id, choice in enumerate(options):    
        print("{}) {}".format(id + 1, choice))  

    # Validation handling
    err_count = 0
    
    while True:
        # i the choice number
        idx = input("Enter choice number: ")    

        # Validation Rules
        try:
            id = int(idx) -1
            
            if id in range(len(options)):
                return id
            
            else:
                print("Not an appropriate choice.")
                err_count +=1
                if err_count > 3:
                        print("Sorry, too many errors \n")
                        break
        except ValueError:
            print("Not an appropriate choice.")
            err_count +=1
            if err_count > 3:
                print("Sorry, too many errors \n")
                break
            
            


def ask_user_continue():
    '''define function for screen break '''
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
    # california_housing.frame.info()
    pd.set_option('display.precision', 4)
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.width' , 400)
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
    print(sample_df.describe())
    print('......California_housing datdaset loaded:')
    
    ask_user_continue()
    
    # print('Plot chart 1')
    ## plotting the eight features as X and MedHouseValue as Y
    for feature in california.feature_names:
        plt.figure(figsize=(16, 9))
        sns.scatterplot(data=sample_df, x=feature,
            y='MedHouseValue', hue='MedHouseValue',
            palette='cool', legend=False)
        # print('8 features')
        
        plt.show()
        print('Chart feature is ', feature)
        ask_user_continue()


def split_train_test(sample_df):
    ''' function to split the sample data into model's train and test data group'''    
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


def load_dataset_diabetes():
    
    import matplotlib.pyplot as plt
    import numpy as np
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
    
    while True:
        try:
            # call func for input,validation and error handling, parsing choice value
            user_choice = let_user_choose(options)   
        except ValueError:
            #ask user if to continute at errors
            ask_user_continue()            
            continue
            
        if (user_choice == 0):                
            load_dataset_cali()
            # ask_user_continue()        
                
        elif (user_choice == 1):
            load_dataset_diabetes()
            # ask_user_continue()
            
            
        elif (user_choice == 2):
            print ("Exiting the program")
       
            sys.exit()
        else:
            
            print('Please choose from the list')
            continue
        
        
        
'''enter please choose dataset 1 2  and exit 3 '''
options = ["California Housing Dataset",  
           "Diabetes",  
           "Exit"]



user_interaction_main()

