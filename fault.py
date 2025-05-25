import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load datasets
old_df = pd.read_csv("old_fault_dataset.csv")
new_df = pd.read_csv("new_fault_dataset.csv")

# Encode fault types
def prepare_data(df):
    df['fault_type'] = df['fault_type'].astype('category')
    df['fault_code'] = df['fault_type'].cat.codes
    X = df[['voltage_A', 'voltage_B', 'voltage_C', 'current_A', 'current_B', 'current_C']]
    y = df['fault_code']
    label_map = dict(enumerate(df['fault_type'].cat.categories))
    return X, y, label_map

# Prepare old dataset
X_old, y_old, label_map = prepare_data(old_df)

# Train model on old dataset
model_old = DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=42)
model_old.fit(X_old, y_old)
y_old_pred = model_old.predict(X_old)
print("Old Dataset Accuracy:", accuracy_score(y_old, y_old_pred))

# Prepare new dataset
X_new, y_new, _ = prepare_data(new_df)
X_temp, X_val, y_temp, y_val = train_test_split(X_new, y_new, test_size=0.70, stratify=y_new, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=1)

# Train on new dataset
model_new = DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=1)
model_new.fit(X_train, y_train)

# Evaluate on test set
y_test_pred = model_new.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report (Test):\n", classification_report(y_test, y_test_pred, target_names=label_map.values()))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))

# Evaluate on validation set
y_val_pred = model_new.predict(X_val)
print("\nValidation Accuracy:", accuracy_score(y_val, y_val_pred))
