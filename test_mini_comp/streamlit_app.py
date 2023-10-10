import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from category_encoders import BaseNEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Title of the page
st.title("Solving the Richter's Predictor: Modeling Earthquake Damage Challange")

# Problem description
# Chapter: Problem Description
st.header("Problem Description")
st.write(
    """We're trying to predict the ordinal variable damage_grade, which represents a level of damage to the building that was hit by the earthquake. There are 3 grades of the damage:
1 represents low damage
2 represents a medium amount of damage
3 represents almost complete destruction"""
)

# make the folder of the file the current working directory
import os

os.chdir(os.path.dirname(__file__))

# Load the data
df_train = pd.read_csv("./data/train_values.csv")
df_labels = pd.read_csv("./data/train_labels.csv")
df_test = pd.read_csv("./data/test_values.csv")

# Chapter: Data Description
st.header("Data Description")
st.write("""Head of the train data""")
st.dataframe(df_train.head())
st.write("""Head of the label data""")
st.dataframe(df_labels.head())
st.write("""Head of the test data""")
st.dataframe(df_test.head())

# Dropping columns
df_train_ = df_train.copy()
df_test_ = df_test.copy()
# Columns to drop
cols_to_drop = [
    "building_id",
    "has_secondary_use_institution",
    "has_secondary_use_school",
    "has_secondary_use_industry",
    "plan_configuration",
    "has_secondary_use_health_post",
    "has_secondary_use_gov_office",
    "has_secondary_use_use_police",
]
df_train = df_train.drop(cols_to_drop, axis=1)
df_test = df_test.drop(cols_to_drop, axis=1)

# removing outliers
cont_cols = df_train.select_dtypes(include=["int64", "float64"]).columns

# exluding the count_families column and age
cont_cols = cont_cols.drop(["count_families", "age"])

# buildings with age > 200 are considered as outliers
outliers = df_train[df_train["age"] > 200].index.tolist()


# check for outliers in the continuous columns automatically
# function that checks for outliers in a column and returns the indices of the outliers
def find_outliers(df, col):
    if df[col].unique().shape[0] < 10:
        return []
    else:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (5 * iqr)
        upper_bound = q3 + (5 * iqr)
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        print(f"Number of outliers in {col}: {len(outliers)}")
        print(f"Percentage of outliers in {col}: {len(outliers)/len(df)*100}%")
        return outliers


# Find the outliers in the continuous columns
# outliers = []
for col in cont_cols:
    outliers.extend(find_outliers(df_train, col))
outliers = list(set(outliers))

print(f"Number of outliers: {len(outliers)}")
print(f"Percentage of outliers: {len(outliers)/len(df_train)*100}%")

# Remove the outliers
outliers = list(set(outliers))
df_train = df_train.drop(outliers, axis=0)
df_labels = df_labels.drop(outliers, axis=0)

# List of all categorical features
cat_features = [col for col in df_train.columns if df_train[col].dtype == "object"]

# BaseN encoding
encoder = BaseNEncoder(cols=cat_features, base=2)
# Mean encoding
# encoder = ce.TargetEncoder(cols=cat_features)
df_train = encoder.fit_transform(df_train, df_labels["damage_grade"])
df_test = encoder.transform(df_test)

# The columns geo_level_1_id, geo_level_2_id, geo_level_3_id are actually categorical features
# We will use mean encoding for these features
geo_features = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
encoder2 = ce.TargetEncoder(cols=geo_features)
df_train = encoder2.fit_transform(df_train, df_labels["damage_grade"])
df_test = encoder2.transform(df_test)

# Scaling the features using StandardScaler but keeping the column names
scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(df_train)
df_train_scaled = pd.DataFrame(df_train_scaled, columns=df_train.columns)
df_test_scaled = scaler.transform(df_test)
df_test_scaled = pd.DataFrame(df_test_scaled, columns=df_test.columns)

df_train = df_train_scaled
df_test = df_test_scaled

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    df_train, df_labels["damage_grade"], test_size=0.2, random_state=42
)

# Labels have to be 0, 1, 2 for XGBoost
y_train_m1 = y_train - 1
y_val_m1 = y_val - 1


# Building the model (XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train_m1)
dval = xgb.DMatrix(X_val, label=y_val_m1)
dtest = xgb.DMatrix(df_test)

# load the model, which was just saved
model_xgb = xgb.Booster()
model_xgb.load_model("model_xgb.json")


y_pred_val = model_xgb.predict(dval)
y_pred_val = y_pred_val + 1
st.write("F1 score on validation set:")
st.write(f1_score(y_val, y_pred_val, average="micro"))

# plot feature importance
ax = xgb.plot_importance(model_xgb)
# the plot output of xgboost is a matplotlib axes object
# we can specify the ax parameter to plot the figure in a specific axes
st.write("Feature importance plot:")
st.pyplot(ax.figure)


# Building a section, where the user can input the data
st.header("Input the data")
st.write(
    """You can input the data of a building and the model will predict the damage grade"""
)

# Using the data and dtypes of the first row of the train data as a template
# to create the input fields
# The data
data = df_train_.iloc[0, :].to_dict()
# The dtypes
dtypes = df_train_.dtypes.to_dict()
# max and min values for the numerical columns
max_values = df_train_.max().to_dict()
min_values = df_train_.min().to_dict()

# Creating the input fields using the data of the first row of the train data as default values
input_fields = {}
for col in df_train_.columns:
    if dtypes[col] == "int64":
        input_fields[col] = st.number_input(
            col, min_value=min_values[col], max_value=max_values[col], value=data[col]
        )
    elif dtypes[col] == "float64":
        input_fields[col] = st.number_input(
            col, min_value=min_values[col], max_value=max_values[col], value=data[col]
        )
    else:
        input_fields[col] = st.selectbox(col, df_train_[col].unique())

# Creating the input dataframe
input_df = pd.DataFrame(input_fields, index=[0])

# Dropping the columns that are not in the train data
input_df = input_df.drop(cols_to_drop, axis=1)

# BaseN encoding
input_df = encoder.transform(input_df)

# The columns geo_level_1_id, geo_level_2_id, geo_level_3_id are actually categorical features
# We will use mean encoding for these features
input_df = encoder2.transform(input_df)

# predict the damage grade for the user input
dinput = xgb.DMatrix(input_df)
y_pred_input = model_xgb.predict(dinput)
y_pred_input = y_pred_input + 1
st.write(f"The predicted damage grade for the user input: {y_pred_input[0]}")
