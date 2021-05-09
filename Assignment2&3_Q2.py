import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.datasets import load_breast_cancer 
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from mlxtend.plotting import plot_decision_regions


#Load the data from CSV file.
data = pd.read_csv("data.csv")
X = data.drop('diagnosis', axis=1)
X = X.iloc[:,:31]

Y = data.iloc[:,1]

lb_make = LabelEncoder()
Y= lb_make.fit_transform(Y) # Convert Y to numarical value for plot_decision_regions

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

#split the dataset into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

#--------------------------Logistic Regression--------------------------
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

features = data.iloc[:, [2,4]].values # Chose 2 features to plot decision boundry
classifier2 = classifier.fit(features, Y)

# Plot decision boundry
plot_decision_regions(np.asarray(features), np.asarray(Y), clf=classifier2, legend=2)
plt.show()

#--------------------------SVM--------------------------
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}  
grid_search = GridSearchCV(svm.SVC(kernel = 'rbf'), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
grid_pred = grid_search.predict(X_test)

# Plot decision boundry
grid_search2 = grid_search.fit(features, Y)
plot_decision_regions(np.asarray(features), np.asarray(Y), clf=grid_search2, legend=2)
plt.show()


print("Logestic", accuracy_score (y_test, y_pred))
print("SVM", accuracy_score (y_test, grid_pred))
# The SVM classifier perform better with very small difference, and this is because the grid search chose the best parameter.
