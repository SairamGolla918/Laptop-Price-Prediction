# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data_set = pd.read_csv('laptop_data.csv')

data_set.head()

data_set.shape

data_set.info()

data_set.duplicated().sum()

data_set.isnull().sum()

data_set.drop(columns=['Unnamed: 0'],inplace=True)

data_set.head()

data_set['Ram'] = data_set['Ram'].str.replace('GB','')
data_set['Weight'] = data_set['Weight'].str.replace('kg',' ')

data_set['Ram'] = data_set['Ram'].astype('int32')

data_set['Weight'] = data_set['Weight'].astype('float32')

data_set.info()

import seaborn as sns

sns.displot(data_set['Price'])

data_set['Company'].value_counts().plot(kind='bar')

sns.barplot(x=data_set['Company'],y=data_set['Price'])
plt.xticks(rotation='vertical')
plt.show()

data_set['TypeName'].value_counts().plot(kind='bar')

sns.barplot(x=data_set['TypeName'],y=data_set['Price'])
plt.xticks(rotation='vertical')
plt.show()

sns.displot(data_set['Inches'])

sns.scatterplot(x=data_set['Inches'],y=data_set['Price'])

data_set['ScreenResolution'].value_counts()

data_set['Touchscreen'] = data_set['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0 )

data_set.head()

data_set.sample(5)

data_set['Touchscreen'].value_counts().plot(kind='bar')

sns.barplot(x=data_set['Touchscreen'],y=data_set['Price'])

data_set['Ips'] = data_set['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

data_set.sample(5)

data_set.head()

data_set['Ips'].value_counts().plot(kind='bar')

sns.barplot(x=data_set['Ips'],y=data_set['Price'])

new = data_set['ScreenResolution'].str.split('x',n=1,expand=True)

data_set['X_res'] = new[0]
data_set['Y_res'] = new[1]

data_set.sample(5)

data_set['X_res'] = data_set['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

data_set.head()

data_set ['X_res'] = data_set['X_res'].astype('int')
data_set['Y_res'] = data_set['Y_res'].astype('int')

data_set.info()

# Convert 'X_res' and 'Y_res' to numeric, handling non-numeric values
data_set['X_res'] = pd.to_numeric(data_set['X_res'], errors='coerce')
data_set['Y_res'] = pd.to_numeric(data_set['Y_res'], errors='coerce')

# Calculate correlations on numeric columns only
data_set.select_dtypes(include=['number']).corr()['Price']

data_set['ppi'] = (((data_set['X_res']**2) + (data_set['Y_res']**2))**0.5/data_set['Inches']).astype('float')

# Convert 'X_res' and 'Y_res' to numeric, handling non-numeric values
data_set['X_res'] = pd.to_numeric(data_set['X_res'], errors='coerce')
data_set['Y_res'] = pd.to_numeric(data_set['Y_res'], errors='coerce')

# Calculate correlations on numeric columns only
data_set.select_dtypes(include=['number']).corr()['Price']

data_set.head()

data_set.drop(columns=['ScreenResolution'],inplace=True)

data_set.head()

data_set.drop(columns=['Inches','X_res','Y_res'],inplace=True)

data_set.head()

data_set['Cpu'].value_counts()

data_set['Cpu Name'] = data_set['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))

data_set.head()

def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

data_set['Cpu brand'] = data_set['Cpu Name'].apply(fetch_processor)

data_set.head()

data_set['Cpu brand'].value_counts().plot(kind='bar')

sns.barplot(x=data_set['Cpu brand'],y=data_set['Price'])
plt.xticks(rotation='vertical')
plt.show()

data_set.drop(columns=['Cpu','Cpu Name'],inplace=True)

data_set.head()

data_set['Ram'].value_counts().plot(kind='bar')

sns.barplot(x=data_set['Ram'],y=data_set['Price'])
plt.xticks(rotation='vertical')
plt.show()

data_set['Memory'].value_counts()

data_set['Memory'] = data_set['Memory'].astype(str).replace('\.0', '', regex=True)
data_set["Memory"] = data_set["Memory"].str.replace('GB', '')
data_set["Memory"] = data_set["Memory"].str.replace('TB', '000')
new = data_set["Memory"].str.split("+", n = 1, expand = True)

data_set["first"]= new[0]
data_set["first"]=data_set["first"].str.strip()

data_set["second"]= new[1]

data_set["Layer1HDD"] = data_set["first"].apply(lambda x: 1 if "HDD" in x else 0)
data_set["Layer1SSD"] = data_set["first"].apply(lambda x: 1 if "SSD" in x else 0)
data_set["Layer1Hybrid"] = data_set["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
data_set["Layer1Flash_Storage"] = data_set["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

# The issue is here. The previous regex was not removing all non-digit characters
data_set["first"] = data_set["first"].str.replace(r"\D", "", regex=True) # Use regex=True for str.replace
data_set["first"] = data_set["first"].astype(int)

data_set["second"].fillna("0", inplace = True)

data_set["Layer2HDD"] = data_set["second"].apply(lambda x: 1 if "HDD" in x else 0)
data_set["Layer2SSD"] = data_set["second"].apply(lambda x: 1 if "SSD" in x else 0)
data_set["Layer2Hybrid"] = data_set["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
data_set["Layer2Flash_Storage"] = data_set["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)


data_set['second'] = data_set['second'].str.replace(r'\D', '', regex=True)


data_set["second"] = data_set["second"].astype(int)

data_set["HDD"]=(data_set["first"]*data_set["Layer1HDD"]+data_set["second"]*data_set["Layer2HDD"])
data_set["SSD"]=(data_set["first"]*data_set["Layer1SSD"]+data_set["second"]*data_set["Layer2SSD"])
data_set["Hybrid"]=(data_set["first"]*data_set["Layer1Hybrid"]+data_set["second"]*data_set["Layer2Hybrid"])
data_set["Flash_Storage"]=(data_set["first"]*data_set["Layer1Flash_Storage"]+data_set["second"]*data_set["Layer2Flash_Storage"])

data_set.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)

data_set.sample(5)

data_set.drop(columns=['Memory'],inplace=True)

data_set.head()

# Check if 'X_res' column exists in the DataFrame
if 'X_res' in data_set.columns:
    # Convert 'X_res' and 'Y_res' to numeric, handling non-numeric values
    data_set['X_res'] = pd.to_numeric(data_set['X_res'], errors='coerce')
    data_set['Y_res'] = pd.to_numeric(data_set['Y_res'], errors='coerce')

    # Calculate correlations on numeric columns only
    correlations = data_set.corr()['Price']
    print(correlations)
else:
    print("Column 'X_res' not found in the DataFrame.")

data_set.drop(columns=['Hybrid','Flash_Storage'],inplace=True)

data_set.head()

data_set['Gpu'].value_counts()

data_set['Gpu brand'] = data_set['Gpu'].apply(lambda x:x.split()[0])

data_set.head()

data_set['Gpu brand'].value_counts()

data_set = data_set[data_set['Gpu brand'] != 'ARM']

sns.barplot(x=data_set['Gpu brand'],y=data_set['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()

data_set.drop(columns=['Gpu'],inplace=True)

data_set.head()

data_set['OpSys'].value_counts()

sns.barplot(x=data_set['OpSys'],y=data_set['Price'])
plt.xticks(rotation='vertical')
plt.show()

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

data_set['os'] = data_set['OpSys'].apply(cat_os)

data_set.head()

data_set.drop(columns=['OpSys'],inplace=True)

sns.barplot(x=data_set['os'],y=data_set['Price'])
plt.xticks(rotation='vertical')
plt.show()

sns.distplot(data_set['Weight'])

sns.scatterplot(x=data_set['Weight'],y=data_set['Price'])

# Convert the 'Company' column to a numerical representation
# (e.g., using label encoding or one-hot encoding) before calculating correlations.
# Example using label encoding:
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# Check if 'Company' column exists before encoding
if 'Company' in data_set.columns:
    data_set['Company_Encoded'] = label_encoder.fit_transform(data_set['Company'])

    # Now calculate the correlations, excluding the original 'Company' column
    data_set.drop(columns=['Company'], inplace=True)  # Remove the original string column

    # Convert 'Company_Encoded' to numeric type
    data_set['Company_Encoded'] = data_set['Company_Encoded'].astype(float)

    # Identify and handle other non-numeric columns
    non_numeric_cols = data_set.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
        if col != 'Price':  # Assuming 'Price' is your target variable and is numeric
            data_set[col] = label_encoder.fit_transform(data_set[col]) # Encode other non-numeric columns
            data_set[col] = data_set[col].astype(float)

    correlations = data_set.corr()['Price']
    print(correlations)
else:
    print("The 'Company' column does not exist in the DataFrame.")

# Select only the numeric columns before calculating correlations
numeric_data = data_set.select_dtypes(include=['number'])

# Calculate correlations on the numeric subset
correlations = numeric_data.corr()
print(correlations)

# Identify and handle remaining non-numeric columns
non_numeric_cols = data_set.select_dtypes(exclude=['number']).columns
for col in non_numeric_cols:
    # Use label encoding or one-hot encoding based on the nature of the column
    data_set[col] = label_encoder.fit_transform(data_set[col])
    data_set[col] = data_set[col].astype(float)  # Convert to numeric type

# Now calculate correlations and generate the heatmap
sns.heatmap(data_set.corr())

sns.distplot(np.log(data_set['Price']))

X = data_set.drop(columns=['Price'])
y = np.log(data_set['Price'])

X

y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)

X_train

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor

X_train

X_train.info()

"""LinearRegression"""

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))

"""RidgeRegression"""

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Ridge(alpha=10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))

"""Random Tree"""

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))

import pickle

pickle.dump(data_set,open('data_set.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))



