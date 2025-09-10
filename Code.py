import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_excel(
    'healthcare_dataset.xlsx',
    engine='openpyxl'
)

# Display the first 5 rows and a summary of the data
print("Initial Data Snapshot:")
print(df.head())
print("\nDataFrame Info:")
df.info()

# Create a copy to avoid modifying the original DataFrame
df_processed = df.copy()

# A. Convert 'Date of Admission' and 'Discharge Date' to datetime objects
df_processed['Date of Admission'] = pd.to_datetime(df_processed['Date of Admission'])
df_processed['Discharge Date'] = pd.to_datetime(df_processed['Discharge Date'])

# B. Handle Missing Values
# Average stay duration for filling missing Discharge Dates
avg_stay_days = (df_processed['Discharge Date'] - df_processed['Date of Admission']).dt.days.mean()
df_processed['Discharge Date'].fillna(
    df_processed['Date of Admission'] + pd.to_timedelta(avg_stay_days, unit='D'),
    inplace=True
)

# Fill missing 'Test Results' with mode
mode_test_results = df_processed['Test Results'].mode()[0]
df_processed['Test Results'].fillna(mode_test_results, inplace=True)

# C. Clean and Convert 'Billing Amount'
df_processed['Billing Amount'] = df_processed['Billing Amount'].astype(str)
df_processed['Billing Amount'] = df_processed['Billing Amount'].str.replace('$', '', regex=False).astype(float)

# D. Encode Categorical Data
# One-hot encode 'Test Results'
df_processed = pd.get_dummies(df_processed, columns=['Test Results'], prefix='Test_Results')

# Label encode 'Gender'
le = LabelEncoder()
df_processed['Gender_encoded'] = le.fit_transform(df_processed['Gender'])

# -------------------- Train-Test Split --------------------
# Example: Suppose target column is 'Billing Amount' (replace with your target)
X = df_processed.drop(columns=['Billing Amount'])   # Features
y = df_processed['Billing Amount']                  # Target

# Split data into train & test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nShapes of Train/Test Split:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
# ----------------------------------------------------------

# Save preprocessed file
excel_file = "healthcare_dataset_preprocessed.xlsx"
df_processed.to_excel(excel_file, index=False, engine="openpyxl")
print(f"Excel saved as: {excel_file}")

# Display the first 5 rows and a summary of the preprocessed data
print("\nProcessed DataFrame Info:")
df_processed.info()
print("\nProcessed Data Snapshot:")
print(df_processed.head())
