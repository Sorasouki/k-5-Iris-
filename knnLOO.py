from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean, sqrt, absolute
import pandas as pd

#read in the data using pandas
df = pd.read_csv('https://github.com/achmatim/data-mining/blob/main/Dataset/iris.csv?raw=true')
pd.set_option('display.max_columns', None)
#check data has been read in properly
df.head()
#check number of rows and columns in dataset
df.shape
#create a dataframe with all training data except the target column
X = df.drop(columns=['Label'])
#check that the target variable has been removed
X.head()
#separate target values
y = df['Label'].values
#view target values
y[0:5]

#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=5)
#train model with cv of kFold equal to the amount of data
kfolds = KFold(n_splits=150, random_state=None)
cv_scores = cross_val_score(knn_cv, X, y, cv=kfolds)
mean_cv_scores = mean(absolute(cv_scores))
error_cv_scores = sqrt(mean(absolute(cv_scores)))
#print each cv score (accuracy) and average them
print('cv_scores mean =', '%.2f'%(mean_cv_scores*100), '%')
print('cv_scores Error mean =', '%.2f'%(100-(mean_cv_scores*100)), '%')
print('cv_scores True Error mean =', '%.2f'%(100-(error_cv_scores*100)), '%')
