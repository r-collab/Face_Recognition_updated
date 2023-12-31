import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

# Read the face measurements from the Excel file into a pandas DataFrame
data = pd.read_excel("face_measurements.xlsx")

# Separate the features (face measurements) and the target variable (name) from the DataFrame
X = data.drop("Name", axis=1)
y = data["Name"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train an SVM classifier on the training data
svm = SVC()
svm.fit(X_train_scaled, y_train)

with open("svm_model.pkl", "wb") as file:
    pickle.dump(svm, file)

with open("scaler_model.pkl", "wb") as file:
    pickle.dump(scaler, file)


# Evaluate the classifier on the testing data
accuracy = svm.score(X_test_scaled, y_test)
print("Accuracy:", accuracy)
