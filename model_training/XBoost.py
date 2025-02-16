import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb

# Step 1: Data Preprocessing (same as before)
def load_csv_files_recursive(folder_path):
    dataframes = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".csv"):
                file_path = os.path.join(root, filename)
                try:
                    if os.path.getsize(file_path) > 0:  # Check if the file is not empty
                        df = pd.read_csv(file_path)
                        if not df.empty:
                            df['source_file'] = filename  # Add a column for the source file
                            df['category'] = os.path.basename(root)  # Add the folder name as the category
                            dataframes.append(df)
                            print(f"Loaded {filename} from {root} with shape {df.shape}")
                        else:
                            print(f"Warning: {filename} is empty and will be skipped.")
                    else:
                        print(f"Warning: {filename} is empty and will be skipped.")
                except pd.errors.EmptyDataError:
                    print(f"Error: {filename} is empty or does not contain valid data.")
                except Exception as e:
                    print(f"Error: Could not process {filename}. Reason: {e}")
            else:
                print(f"Skipped non-CSV file: {filename} in {root}")
    return dataframes

# Load the data
folder_path = 'D:\\Srihith - Copy\\processed_data'
dataframes = load_csv_files_recursive(folder_path)

# Check if any dataframes are loaded
if len(dataframes) == 0:
    raise ValueError("No valid CSV files found in the folder.")

# Combine all dataframes into a single dataframe
combined_df = pd.concat(dataframes, ignore_index=True)

# Data Preprocessing (e.g., handling missing values, removing unnecessary columns)
combined_df.fillna('', inplace=True)

# Step 2: Feature Extraction
# Example feature: Number of rows and columns in the table
combined_df['num_rows'] = combined_df.apply(lambda row: len(row), axis=1)
combined_df['num_cols'] = combined_df.apply(lambda row: len(row.index), axis=1)

# Example feature: Presence of specific keywords
keywords = ['income', 'balance', 'cash', 'flow', 'notes']
vectorizer = CountVectorizer(vocabulary=keywords)
keyword_matrix = vectorizer.fit_transform(combined_df.apply(lambda x: ' '.join(x.astype(str)), axis=1))

# Combine features into a feature matrix
features = np.hstack([combined_df[['num_rows', 'num_cols']].values, keyword_matrix.toarray()])

# Labels
labels = combined_df['category'].values

# Step 3: Model Selection and Training
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the XGBoost Classifier
xgb_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_classifier.fit(X_train, y_train)

# Step 4: Model Evaluation
# Predict on the test set
y_pred = xgb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
print(classification_report(y_test, y_pred, target_names=['Income Statements', 'Balance Sheets', 'Cash Flows', 'Notes', 'Others']))

# Save the model (if needed)
import joblib
joblib.dump(xgb_classifier, 'financial_statement_xgb_classifier.pkl')

# Load the model (if needed)
# xgb_classifier = joblib.load('financial_statement_xgb_classifier.pkl')
