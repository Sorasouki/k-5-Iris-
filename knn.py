import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import sqrt, mean, absolute
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

#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 5)
# Fit the classifier to the data
knn.fit(X_train,y_train)
#show first 5 model predictions on the test data
knn.predict(X_test)[0:5]
#check accuracy of our model on the test data
acc = (knn.score(X_test, y_test))
error_acc = sqrt(absolute(acc))
print('Accuracy =', '%.2f'%(acc*100), '%')
print('Error =', '%.2f'%(100-(acc*100)), '%')
print('True Error Probability =', '%.2f'%(100-(error_acc*100)), '%')
