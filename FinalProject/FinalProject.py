# import libraries
import time
import pandas as pd
import numpy as np
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

############################## Functions ####################################################################################################

# Removing outliers


def remove_outliers(df, column_name, sd_multiplier=3):

 # Compute the mean and standard deviation of the column
    column_mean = df[column_name].mean()
    column_std = df[column_name].std()

    # Compute the lower and upper bounds for outlier removal
    lower_bound = column_mean - (sd_multiplier * column_std)
    upper_bound = column_mean + (sd_multiplier * column_std)

    # Create a copy of the DataFrame with outliers and zeros removed
    cleaned_df = df.loc[(df[column_name] > lower_bound) & (
        df[column_name] < upper_bound) & (df[column_name] != 0)].copy()

    return cleaned_df

# Dataframe to Bunch


def df_to_bunch(df):
    # Get the feature names
    feature_names = df.columns.tolist()

    # Get the target variable name
    target_name = feature_names.pop(-1)

    # Get the feature data and target data
    data = df[feature_names].values
    target = df[target_name].values

    # Create a Bunch object
    bunch = Bunch(data=data, target=target,
                  feature_names=feature_names, target_names=target_name)

    return bunch


################################### Load the dataset ########################################################################################
t0 = time.perf_counter()

temp_data = pd.read_csv(
    '../Temp_history_final.csv')
load_data = pd.read_csv(
    '../Load_history_final.csv')

# Melt the temperature columns into a new DataFrame
temp_df = pd.melt(
    temp_data,
    id_vars=['station_id', 'year', 'month', 'day'],
    value_vars=['h{}'.format(i) for i in range(1, 25)],
    var_name='hour',
    value_name='temp'
)
temp_df['hour'] = temp_df['hour'].str.slice(start=1).astype(int)

# Melt the load columns into a new DataFrame
load_df = pd.melt(
    load_data,
    id_vars=['zone_id', 'year', 'month', 'day'],
    value_vars=['h{}'.format(i) for i in range(1, 25)],
    var_name='hour',
    value_name='load'
)
load_df['hour'] = load_df['hour'].str.slice(start=1).astype(int)

zone_to_stat = {1: 7, 2: 7, 3: 9, 4: 9, 5: 2, 6: 9, 7: 9, 8: 10, 9: 8, 10: 11,
                11: 8, 12: 6, 13: 3, 14: 8, 15: 10, 16: 4, 17: 9, 18: 4, 19: 6, 20: 9}

# Create array from temperature dataframe
load_arr = load_df.to_numpy()

# Change the station values for the corresponding zones
zone_arr = load_arr[:, 0]
mapped_list = [zone_to_stat[val] for val in zone_arr]
mapped_array = np.array(mapped_list)
new_arr = np.hstack(
    (load_arr[:, :1], mapped_array.reshape(-1, 1), load_arr[:, 1:]))
zs_load_df = pd.DataFrame(new_arr, columns=[
                          'zone_id', 'station_id', 'year', 'month', 'day', 'hour', 'load'])

# Merge both dataframes to a single one containing temperature and load values
zone_merged_df = pd.merge(zs_load_df, temp_df, on=['year', 'month', 'day', 'hour', 'station_id'], how='left')[
    ['zone_id', 'station_id', 'year', 'month', 'day', 'hour', 'temp', 'load']]

zone_merged_df_t = remove_outliers(zone_merged_df, 'temp')
zone_merged_df_t = remove_outliers(zone_merged_df, 'load')

# Data splitting
data = df_to_bunch(zone_merged_df_t)
X = data.data
y = data.target

# Split the data into training/validation (80%) and test (20%) sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3, shuffle=True)

# Split the training/validation set into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=3, shuffle=True)

################################## Training DecisionTreeRegressor ###########################################################################

best_params_DTR = {'max_depth': None, 'max_leaf_nodes': None, 'min_samples_leaf': 3,
                   'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'splitter': 'best'}

best_regr = DecisionTreeRegressor(**best_params_DTR)
print(f'Refressor used: {type(best_regr)}')

# Train regressor
best_regr.fit(X_train, y_train)

# Obtain scores for data splits
train_score_DTR = best_regr.score(X_train, y_train)
val_score_DTR = best_regr.score(X_val, y_val)
test_score_DTR = best_regr.score(X_test, y_test)
print(
    f"Training, Validation & Test scores for DTR: {train_score_DTR}, {val_score_DTR} & {test_score_DTR}")

# Make predictions and obtain accuracy scores
y_test_pred = best_regr.predict(X_test)

test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print('Test R2:', test_r2)
print('Test MSE:', test_mse)

# Load prediction
df_t = zone_merged_df[(zone_merged_df['year'] == 2008) & (
    zone_merged_df['month'] == 6) & (zone_merged_df['day'] <= 7)]
df_t = df_t.sort_values(['zone_id', 'hour'])


array_t = df_t.to_numpy()
X = array_t[:, :-1]
y_pred = best_regr.predict(X)
y_pred = y_pred.reshape(-1, 1)
array_t = np.append(X, y_pred, axis=1)

df = pd.DataFrame(array_t[:, [0, 2, 3, 4, 5, 7]], columns=[
                  'zone_id', 'year', 'month', 'day', 'hour', 'load'])

# pivot the dataframe to convert hour column to h1 to h24 columns
pivoted_df = df.pivot_table(index=['zone_id', 'year', 'month', 'day'],
                            columns='hour',
                            values='load',
                            aggfunc='first').reset_index()

# rename the h1 to h24 columns
pivoted_df.columns = ['zone_id', 'year', 'month',
                      'day'] + ['h{}'.format(h) for h in range(1, 25)]

pivoted_df.to_csv('Load_prediction.csv', index=False)

t1 = time.perf_counter()
print(f"Total running time: {t1 - t0:0.4f} seconds")


print('output file with name "Load_prediction.csv" was created')
