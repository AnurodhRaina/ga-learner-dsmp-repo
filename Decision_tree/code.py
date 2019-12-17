# --------------
#Importing header files

import pandas as pd
from sklearn.model_selection import train_test_split as tts


# Code starts here

data= pd.read_csv(path)

X= data.drop(['customer.id','paid.back.loan'],1)
y=data['paid.back.loan']

X_train, X_test, y_train, y_test = tts(X,y,random_state=0,test_size=0.3)
# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here

import pandas as pd
from sklearn.model_selection import train_test_split as tts


# Code starts here


fully_paid = y_train.value_counts()
plt.figure()
fully_paid.plot(kind='bar')
# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here
X_train['int.rate'] = X_train['int.rate'].str.replace('%','').astype(float)
X_train['int.rate'] = X_train['int.rate']/100

X_test['int.rate'] = X_test['int.rate'].str.replace('%','').astype(float)
X_test['int.rate'] = X_test['int.rate']/100

num_df = X_train.select_dtypes(include = np.number)
cat_df = X_train.select_dtypes(exclude = np.number)




# Code ends here


# --------------
#Importing header files
import seaborn as sns


# Code starts here


# Code ends 
cols = list(num_df)

fig, axes = plt.subplots(nrows =9, ncols= 1)
for i in range(1,9):
    sns.boxplot(x=y_train, y=num_df[cols[i]], ax=axes[i])


# --------------
# Code starts here



# Code ends here
cols= list(cat_df)
fig, axes = plt.subplots(nrows = 2, ncols= 2)

for i in range (0,2):
  for j in range(0,2):
    sns.countplot(x=X_train[cols[i*2+j]], hue=y_train, ax=axes[i,j])



# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
# Code starts here


for i in list(cat_df):
    X_train[i].fillna('NA')
    le = LabelEncoder()
    X_train[i] = le.fit_transform(X_train[i])

    X_test[i].fillna('NA')
    le = LabelEncoder()
    X_test[i] = le.fit_transform(X_test[i])

#y_test = y_test.str.replace('No',0)
y_train.replace({'No':0,'Yes':1},inplace=True)
y_test.replace({'No':0,'Yes':1},inplace=True)
# Code ends here

from sklearn.metrics import accuracy_score
model = DecisionTreeClassifier(random_state = 0)
model.fit(X_train, y_train)
y_preds = model.predict(X_test)
acc= accuracy_score(y_test, y_preds)


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

# Code starts here
model_2 = DecisionTreeClassifier(random_state =0)
p_tree = GridSearchCV(estimator=model_2, param_grid=parameter_grid, cv=5)
p_tree.fit(X_train,y_train)

# Code ends here
ypreds2 = p_tree.predict(X_test)
acc_2 = accuracy_score(y_test, ypreds2)

acc_2


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here

dot_data = export_graphviz(decision_tree=p_tree.best_estimator_, out_file=None, feature_names=X.columns, filled = True, class_names=['loan_paid_back_yes','loan_paid_back_no'])

graph_big=pydotplus.graph_from_dot_data(dot_data)



# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here


