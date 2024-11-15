import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
import joblib
import os
import numpy as np
import xgboost as xgb

# Đảm bảo thư mục 'model' tồn tại
if not os.path.exists('model'):
    os.makedirs('model')

# Đọc dữ liệu từ file CSV
dataset = pd.read_csv('data/train.csv')

# Chuyển đổi giá trị cột 'Sex' thành số
def convert_sex(sex):
    if sex == 'F':
        return 1
    elif sex == 'M':
        return 0
    elif sex == 'I':
        return 2
    else:
        return -1  # Giá trị không hợp lệ

dataset['Sex'] = dataset['Sex'].apply(convert_sex)

dataset['Diameter_to_Height_Ratio'] = dataset['Diameter'] / dataset['Height']
dataset['Combined_Whole_Weight'] = dataset['Whole weight'] + dataset['Whole weight.1'] + dataset['Whole weight.2']
dataset['Diameter_Length_Product'] = dataset['Diameter'] * dataset['Length']

dataset['Shell_Volume'] = (4/3) * 3.14 * (dataset['Diameter'] / 2)**2 * dataset['Height']
dataset['Shell_Surface_Area'] = 4 * 3.14 * (dataset['Diameter'] / 2)**2
dataset['Shell_Density'] = dataset['Shell weight'] / dataset['Shell_Volume']
dataset['Shell_Thickness'] = dataset['Height'] - dataset['Diameter']
dataset['Shell_Shape_Index'] = dataset['Shell_Surface_Area'] / dataset['Shell_Volume']
dataset['Length_to_Height_Ratio'] = dataset['Length'] / dataset['Height']

# ////////////////////////////////////////////////////////////////////////////////////////////////////////
unique_counts = dataset.nunique()
#Threshold to distinguish continous and categorical
threshold = 12
continuous_vars_temp = unique_counts[unique_counts > threshold].index.tolist()
#categorical_vars = unique_counts[unique_counts <= threshold].index.tolist()
if 'id' in continuous_vars_temp:
    continuous_vars_temp.remove('id')
numerical_columns = dataset.select_dtypes(include='number').columns
def remove_outliers_replace(data, columns, threshold=1.5):
    data_no_outliers = data.copy()

    for column in columns:
        Q1 = data_no_outliers[column].quantile(0.25)
        Q3 = data_no_outliers[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        is_outlier = (data_no_outliers[column] < lower_bound) | (data_no_outliers[column] > upper_bound)

        if data_no_outliers[column].dtype == 'O':  # Categorical column
            median_value = data_no_outliers.loc[~is_outlier, column].mode().iloc[0]
            data_no_outliers.loc[is_outlier, column] = median_value
        else:  # Numerical column
            mean_value = data_no_outliers.loc[~is_outlier, column].mean()
            data_no_outliers.loc[is_outlier, column] = mean_value

    return data_no_outliers

columns_to_remove_outliers_replace = continuous_vars_temp
dataset = remove_outliers_replace(dataset, columns_to_remove_outliers_replace)

# /////////////////////////////////////////////////////////////////////////////////////////////////////////

dataset.drop(columns='id',axis = 1,inplace = True)
X = dataset.drop(columns='Rings', axis=1).values
y = dataset['Rings'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)


best_params_cat ={'learning_rate': 0.07855075323884125, 'n_estimators': 489, 'max_depth': 7,
                  'subsample': 0.8569934338945397, 'colsample_bylevel': 0.8150591618201379,
                  'reg_lambda': 0.4264547280178772}
model_cat = CatBoostRegressor(**best_params_cat)
model_cat.fit(X, y)

best_params = {'learning_rate': 0.02708319027879099, 'n_estimators': 495, 'reg_alpha': 0.14219650343206225,
               'reg_lambda': 0.5045620145662986, 'max_depth': 12, 'subsample': 0.9537603851451735,
               'colsample_bytree': 0.7819555259366398, 'min_child_weight': 1.03921704772634}
model_xgb = XGBRegressor(**best_params)
model_xgb.fit(X, y)

# Lưu mô hình đã huấn luyện
joblib.dump(model_cat, 'model/abalone_model_cat.pkl')
joblib.dump(model_xgb, 'model/abalone_model_xgb.pkl')

print("Mô hình Catboost đã được lưu thành công vào file 'model/abalone_model_cat.pkl'")
print("Mô hình Catboost đã được lưu thành công vào file 'model/abalone_model_xgb.pkl'")


