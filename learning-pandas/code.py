# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 
bank=pd.read_csv(path)
categorical_var=bank.select_dtypes(include= 'object')

print(categorical_var)

numerical_var = bank.select_dtypes(include='number')
 
print(numerical_var)

# code starts here






# code ends here


# --------------
# code starts here

# load the dataset and drop the Loan_ID
banks= bank.drop(columns='Loan_ID')


# check  all the missing values filled.

print(banks.isnull().sum())

# apply mode 

bank_mode = banks.mode().iloc[0]
print(bank_mode)
# Fill the missing values with 

banks.fillna(bank_mode, inplace=True)

# check again all the missing values filled.

print(banks.isnull().sum())





#code ends here


# --------------
# Code starts here



# code ends here

avg_loan_amount=pd.pivot_table(banks, index=['Gender','Married','Self_Employed'],values='LoanAmount')

print(avg_loan_amount)


# --------------
# code starts here


# code ends here
print(banks.shape)
loanapp = (banks['Self_Employed']=='Yes') & (banks['Loan_Status']=='Y')
loan_approved_se = len(banks[loanapp])
###
loanapp2 = (banks['Self_Employed']=='No') & (banks['Loan_Status']=='Y')
loan_approved_nse = len(banks[loanapp2])

percentage_se = (loan_approved_se/614)*100
percentage_nse = (loan_approved_nse/614)*100


# --------------
# code starts here

loan_term=banks['Loan_Amount_Term'].apply(lambda x: x/12)
big_loan_term = len(banks[loan_term>=25])

print(big_loan_term)


#lambda x: x/30
# code ends here


# --------------
# code starts here

loan_groupby = banks.groupby('Loan_Status')
loan_groupby = loan_groupby[['ApplicantIncome','Credit_History']]
mean_values=loan_groupby.mean()



# code ends here


