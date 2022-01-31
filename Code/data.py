import numpy as np
import pandas as pd

DataSet = pd.read_csv('train.csv')
DataSet = DataSet.drop(['PassengerId','Name','Ticket','Cabin','Survived'], axis = 1)
numCols = DataSet.loc[:, DataSet.select_dtypes(['int64','float64']).columns].copy()
objCols = DataSet.loc[:, DataSet.select_dtypes(['object']).columns].copy()

nan_to = 0
Sex_map = {'male': 1, 'female': 0, np.nan: nan_to}
Embarked_map = {'S': 2, 'C': 1, 'Q': 0, np.nan: nan_to}
objCols['Sex'] = objCols['Sex'].map(Sex_map)
objCols['Embarked'] = objCols['Embarked'].map(Embarked_map)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(numCols)
numCols = pd.DataFrame(imputer.transform(numCols), columns = numCols.columns)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
numCols = pd.DataFrame(sc.fit_transform(numCols), columns = numCols.columns)
objCols == pd.DataFrame(objCols)
data = pd.concat([numCols, objCols], axis = 1)

np.save('data.npy', data)

data.head() 