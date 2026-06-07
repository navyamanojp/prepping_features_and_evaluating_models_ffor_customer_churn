import pandas as pd
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)
print("dataset info :")
print(df.info())
print("dataset preview :")
print(df.head())
categorical_features=df.select_dtypes(include=['object']).columns
print("categorical features :")
print(categorical_features)
numerical_features=df.select_dtypes(include=['int64','float64']).columns
print("numerical features :")   
print(numerical_features)
print("categorical features summary :")
for col in categorical_features:
    print(f"{col}:", df[col].value_counts())
print("numerical features summary :")
print(df[numerical_features].describe())