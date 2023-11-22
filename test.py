import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the phishing dataset
df = pd.read_csv("dataset.csv")

# Convert the 'url' and 'domain' columns into numerical ones using label encoding
labelencoder = LabelEncoder()
df['URL'] = labelencoder.fit_transform(df['URL'])
df['Label'] = labelencoder.fit_transform(df['Label'])

# Split the dataset into training and testing sets
X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict the results for the testing dataset
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
