from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from datetime import date
from sklearn.preprocessing import OrdinalEncoder

# X = Features (your df2 without target)
import pandas as pd
df2 = pd.read_csv('claimsdata_1.csv')
df2['full_name']=df2['first_name']+df2['last_name']
df2.drop(columns=['first_name','last_name','ssn','policy_number','policy_start_date','policy_expiry_date','claim_date','hospital_name',	'hospital_location'	,'user_address'],inplace=True)
if 'Unamed: 0' in df2.columns:
    df2.drop(columns=['Unnamed: 0'],inplace=True)


# Convert the string DOB to datetime
# df2['dob'] = pd.to_datetime(df2['dob'], format='%d-%m-%y')

# # Calculate age accurately


# df2['age'] = df2['dob'].apply(lambda x: date.today().year - x.year)
# df2.drop('dob', axis=1, inplace=True)


# Drop dob column if not needed
df2.drop('dob', axis=1, inplace=True)


encoders = {}

for col in df2.columns:
    if df2[col].dtype == 'object':
        encoder = OrdinalEncoder()
        df2[col] = encoder.fit_transform(df2[[col]])
        encoders[col] = encoder  # Save the encoder for each column

# df2[['full_name', 'age']].head()
X = df2.drop('fraud_flag', axis=1)  # assuming your anomaly column is the label
y = df2['fraud_flag']

# Train-test split


# KNN Classifier (supervised)
knn = KNeighborsClassifier(n_neighbors=5)
print('input columns are')
print(X.info())
knn.fit(X, y)

# Evaluation
# y_pred = knn.predict(X_test)
# print(classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(knn, 'model/knn_classifier.pkl')
joblib.dump(encoders, 'model/encoders.pkl')
