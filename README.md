# PRODIGY_DS_03
Build decision tree classifier to predict customer purchase based on demographic and behavioral data using Bank Marketing dataset.




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
url = r'D:datast.csv'
df = pd.read_csv(url, delimiter=';')


# Step 2: Select relevant columns
selected_columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
df = df[selected_columns]

# Step 3: Preprocess the data
# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'], drop_first=True)

# Convert 'y' (target variable) to binary values (0 or 1)
df['y'] = df['y'].map({'no': 0, 'yes': 1})

# Step 4: Split the data into features (X) and the target variable (y)
X = df.drop('y', axis=1)
y = df['y']

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = clf.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
     
