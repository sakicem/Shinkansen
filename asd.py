import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_data():
    travelData_train = pd.read_csv("Traveldata_train.csv")
    travelData_test = pd.read_csv("Traveldata_test.csv")
    surveyData_train = pd.read_csv("Surveydata_train.csv")
    surveyData_test = pd.read_csv("Surveydata_test.csv")

    df_train = pd.merge(travelData_train, surveyData_train, on='ID')
    df_test = pd.merge(travelData_test, surveyData_test, on='ID')

    def handle_missing_data(df):
        # Fill missing values with the mean of each numeric column
        numeric_columns = df.select_dtypes(include='number').columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

        # Fill missing values with the most common element in each categorical column
        categorical_columns = df.select_dtypes(include='object').columns
        for column in categorical_columns:
            most_common = df[column].mode().values[0]
            df[column].fillna(most_common, inplace=True)

    handle_missing_data(df_train)
    handle_missing_data(df_test)

    # Concatenate train and test datasets for encoding
    df_combined = pd.concat([df_train, df_test])

    # Perform one-hot encoding on categorical columns
    encoder = LabelEncoder()
    categorical_columns = df_combined.select_dtypes(include='object').columns
    for col in categorical_columns:
        df_combined[col] = encoder.fit_transform(df_combined[col])

    # Split the combined dataset back into train and test
    df_train = df_combined[:len(df_train)]
    df_test = df_combined[len(df_train):]

    return df_train, df_test