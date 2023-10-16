import pandas as pd
import numpy as np

"""
1) Milyen adatok kellenek? 
    usage (y)
    weather 
    workday
    pricing
    tourism
Az első háromról van adatunk csak
2) Clean data
3) Distibution of y: log transform/outlier removal/...
4) Train test split
5) Evaluation metric: MAPE, MSE, MAD, RMSLE (Root Mean Squared Logged Error, hasonló mint a MAPE, százalékos 
    értékként is felfogható, ami könnyen interpretálható)
    Kell benchmarkot választani hozzá 
Mit vizsgáljunk? 
    következő nap
    óránként
    évre előre
"""

bike_data=pd.read_csv('bike_sample.csv',sep=',')
print(bike_data.head())

# train-validation split on numeric features
from sklearn.model_selection import train_test_split

feature_matrix = bike_data.drop(columns=["count"]).select_dtypes(include=np.number)
label = bike_data["count"]
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, label, test_size=0.2, random_state=20231016)

# define loss function
def calculateRMSLE(prediction, y_obs):
    return np.sqrt(np.mean(np.power(np.log1p(np.where(0>prediction,0,prediction)) - np.log1p(y_obs), 2)))

# estimate benchmark model
benchmark = np.mean(y_train)
benchmark_result = ['Benchmark', calculateRMSLE(benchmark,y_train), calculateRMSLE(benchmark,y_test)]
print(benchmark_result)

# collect results into a DataFrame
result_columns = ['Model', 'Train', 'Test']
pd.DataFrame([benchmark_result], columns=result_columns)

# model #1: group averages by season, holiday and workingday
from sklearn.linear_model import LinearRegression

# create dummy features
features1 = ['season', 'holiday', 'workingday']
feature_matrix1 = pd.get_dummies(X_train[features1], columns=features1, drop_first=True)

# estimate the model
model1 = LinearRegression()
model1.fit(feature_matrix1, y_train)

# predict on train and test
prediction1_train = model1.predict(feature_matrix1)
prediction1_test = model1.predict(pd.get_dummies(X_test[features1], columns=features1, drop_first=True))

# evaluate model #1
group_avg_result = ['Group-avg', calculateRMSLE(prediction1_train,y_train), calculateRMSLE(prediction1_test,y_test)]

# Model #2: Group averages with weather
dummy_features = ['season', 'holiday', 'workingday', 'weather']
numeric_features = ['temp', 'atemp', 'humidity', 'windspeed']

def createFeatureMatrix2(X):
    return pd.get_dummies(X[dummy_features + numeric_features], columns=dummy_features, drop_first=True)

group_avg_with_weather = LinearRegression().fit(createFeatureMatrix2(X_train), y_train)
train_error = calculateRMSLE(group_avg_with_weather.predict(createFeatureMatrix2(X_train)), y_train)
test_error = calculateRMSLE(group_avg_with_weather.predict(createFeatureMatrix2(X_test)), y_test)

group_avg_with_weather_result = ['Group-avg with weather', train_error, test_error]

results = pd.DataFrame([benchmark_result, group_avg_result, group_avg_with_weather_result], columns=result_columns)

# Model #3: very flexible linear with polynomial features
from sklearn.preprocessing import PolynomialFeatures

def createFeatureMatrix3(X):
    poly_features = PolynomialFeatures(degree=4, include_bias=False)
    X_features = pd.get_dummies(X[dummy_features + numeric_features], columns=dummy_features, drop_first=True)
    return poly_features.fit_transform(X_features)

flexible_linear = LinearRegression().fit(createFeatureMatrix3(X_train), y_train)
train_error = calculateRMSLE(flexible_linear.predict(createFeatureMatrix3(X_train)), y_train)
test_error = calculateRMSLE(flexible_linear.predict(createFeatureMatrix3(X_test)), y_test)

flexible_linear_result = ['Flexible linear', train_error, test_error]
results.loc[len(results)] = flexible_linear_result

# Model #4: improve with LASSO
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler

# Create feature matrix for LASSO

min_max_scaler = MinMaxScaler().set_output(transform='pandas')
min_max_scaler.fit(X_train[numeric_features])

def createFeatureMatrix4(X,scaler=min_max_scaler):
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_scaled_numeric_features = scaler.transform(X[numeric_features])
    X_dummy_features = pd.get_dummies(X[dummy_features + numeric_features], columns=dummy_features, drop_first=True)
    return poly_features.fit_transform(pd.concat([X_scaled_numeric_features, X_dummy_features], axis=1))

lasso_model = LassoCV().fit(createFeatureMatrix4(X_train), y_train)
train_error = calculateRMSLE(lasso_model.predict(createFeatureMatrix4(X_train)), y_train)
test_error = calculateRMSLE(lasso_model.predict(createFeatureMatrix4(X_test)), y_test)

lasso_model_result = ['Flexible LASSO', train_error, test_error]
results.loc[len(results)] = lasso_model_result

# Model 5: Regression tree

from sklearn import tree

fitted_tree = tree.DecisionTreeRegressor(max_depth=5).fit(X_train, y_train) # max_depth can be determined by cross-validation

train_error = calculateRMSLE(fitted_tree.predict(X_train), y_train)
test_error = calculateRMSLE(fitted_tree.predict(X_test), y_test)

tree_result = ['Tree', train_error, test_error]
results.loc[len(results)] = tree_result

print(tree.export_text(fitted_tree, feature_names = list(X_train.columns)))

# Model 6: Realistic tree
feature_matrix = bike_data.drop(columns=["count","casual","registered"]).select_dtypes(include=np.number)
label = bike_data["count"]
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, label, test_size=0.2, random_state=20231016)

fitted_tree = tree.DecisionTreeRegressor(max_depth=5).fit(X_train, y_train) # max_depth can be determined by cross-validation

train_error = calculateRMSLE(fitted_tree.predict(X_train), y_train)
test_error = calculateRMSLE(fitted_tree.predict(X_test), y_test)

tree_result = ['Realistic Tree', train_error, test_error]
results.loc[len(results)] = tree_result

# Improve the model with diagnostics

import matplotlib.pyplot as plt

linear_predictions = group_avg_with_weather.predict(createFeatureMatrix2(X_test))
lasso_predictions = lasso_model.predict(createFeatureMatrix4(X_test))
tree_predictions = fitted_tree.predict(X_test)

plt.scatter(y_test, linear_predictions, label='Linear', alpha=0.5)
plt.scatter(y_test, lasso_predictions, label='Lasso', alpha=0.5)
plt.scatter(y_test, tree_predictions, label='Tree', alpha=0.5)
plt.axline((1, 1), slope=1, linestyle='dashed', color='red')
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.legend()
plt.show()

print(bike_data[bike_data['count'] < 10]) # éjszaka nem használják a biciklit, vegyünk bele az időt is a változók közé

# Feature engerineering

def exctractFeatures(df_with_datetime):
    df_with_datetime['datetime']=pd.to_datetime(df_with_datetime['datetime'],utc=True)
    df_with_datetime['year']=df_with_datetime['datetime'].dt.year
    df_with_datetime['month']=df_with_datetime['datetime'].dt.month
    df_with_datetime['day']=df_with_datetime['datetime'].dt.day
    df_with_datetime['hour']=df_with_datetime['datetime'].dt.hour
    df_with_datetime['dayofweek']=df_with_datetime['datetime'].dt.dayofweek

exctractFeatures(bike_data)

feature_matrix = bike_data.drop(columns=["count","casual","registered"]).select_dtypes(include=np.number)
label = bike_data["count"]
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, label, test_size=0.2, random_state=20231016)

dummy_features2 = ['season', 'holiday', 'workingday', 'weather', 'year', 'month', 'day', 'hour', 'dayofweek']

#Linear model with feature engineering

def createFeatureMatrixFE(X):
    return pd.get_dummies(X, columns=dummy_features2, drop_first=True)

linear_FE = LinearRegression().fit(createFeatureMatrixFE(X_train), y_train)
train_error = calculateRMSLE(linear_FE.predict(createFeatureMatrixFE(X_train)), y_train)
test_error = calculateRMSLE(linear_FE.predict(createFeatureMatrixFE(X_test)), y_test)

linear_FE_result = ['Feature engineered linear', train_error, test_error]
results.loc[len(results)] = linear_FE_result

# LASSO with feature engineering

numeric_features = ['temp', 'atemp', 'humidity', 'windspeed', 'year', 'month', 'day', 'hour', 'dayofweek']
min_max_scaler.fit(X_train[numeric_features])

def createFeatureMatrixFE2(X,scaler=min_max_scaler):
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_scaled_numeric_features = scaler.transform(X[numeric_features])
    X_dummy_features = pd.get_dummies(X[dummy_features + numeric_features], columns=dummy_features2, drop_first=True)
    return poly_features.fit_transform(pd.concat([X_scaled_numeric_features, X_dummy_features], axis=1))
"""
lasso_model = LassoCV().fit(createFeatureMatrixFE2(X_train), y_train)
train_error = calculateRMSLE(lasso_model.predict(createFeatureMatrix4(X_train)), y_train)
test_error = calculateRMSLE(lasso_model.predict(createFeatureMatrix4(X_test)), y_test)

lasso_model_result = ['Feature engineered LASSO', train_error, test_error]
results.loc[len(results)] = lasso_model_result
print(results)
"""
# Tree with feature engineering
fitted_tree = tree.DecisionTreeRegressor(max_depth=5).fit(X_train, y_train) # max_depth can be determined by cross-validation

train_error = calculateRMSLE(fitted_tree.predict(X_train), y_train)
test_error = calculateRMSLE(fitted_tree.predict(X_test), y_test)

tree_result = ['Feature engineering Tree', train_error, test_error]
results.loc[len(results)] = tree_result
print(results)