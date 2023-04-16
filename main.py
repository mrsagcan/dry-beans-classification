import numpy as np
import pandas as panda
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

df = panda.read_excel("Dry_Bean_Dataset.xlsx")

#remove duplicates
duplicates = df[df.duplicated()]
df = df.drop_duplicates()
df = df.reset_index(drop=True)

#
# Check for missing values
print(df.isnull().sum())

# Fill in missing values with the mean value only for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Replace missing values with "Unknown" only in the "Class" column
df['Class'] = df['Class'].fillna('Unknown')

# Scale and normalize the numeric columns
scaler = StandardScaler()
normalized = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df[numeric_cols] = normalized.fit_transform(df[numeric_cols])

# create a OneHotEncoder object
ohe = OneHotEncoder()

# fit and transform the "Class" variable
class_ohe = ohe.fit_transform(df[['Class']])

# add the encoded columns to the dataframe
df = df.join(panda.DataFrame(class_ohe.toarray(), columns=ohe.get_feature_names_out(['Class'])))

#Split the dataset into training and testing sets.
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Export the cleaned dataframe to a new excel file
df.to_excel('efetestres.xlsx', index=False)