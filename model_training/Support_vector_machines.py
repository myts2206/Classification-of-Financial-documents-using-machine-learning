import os
from bs4 import BeautifulSoup
import pandas as pd
import chardet
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec
import numpy as np
import pickle

nltk.download('stopwords')
nltk.download('wordnet')

# Define the directories
input_dir = 'D:/Srihith/data'
output_base_dir = 'processed_data'

# Create the output base directory if it doesn't exist
os.makedirs(output_base_dir, exist_ok=True)

def extract_text_from_html(file_path):
    """Extract text data from HTML file tables."""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
        soup = BeautifulSoup(file, 'html.parser')
        text_data = []
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                row_data = [col.get_text(strip=True) for col in cols]
                text_data.append(row_data)
                
        return text_data

def categorize_table(table_data):
    """Categorize table data into predefined categories using regex and keyword matching."""
    table_text = ' '.join([' '.join(row) for row in table_data]).lower()
    
    # Define regex patterns and keywords for each category
    patterns = {
        'Income Statements': [
            r'income statement', r'profit and loss', r'p&l', r'revenue', r'expense'
        ],
        'Balance Sheets': [
            r'balance sheet', r'assets', r'liabilities', r'equity'
        ],
        'Cash Flows': [
            r'cash flow', r'cash flows', r'operating activities', r'investing activities', r'financing activities'
        ],
        'Notes': [
            r'notes', r'footnotes', r'additional information', r'disclosures'
        ],
        'Others': []  # Any table not matching the above categories
    }
    
    for category, regex_list in patterns.items():
        for regex in regex_list:
            if re.search(regex, table_text):
                return category
    
    return 'Others'

def clean_text(text):
    """Clean the input text by removing stop words, applying lemmatization, and handling punctuation."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]
    return ' '.join(tokens)

# Process each file in the input directory
all_data = []
labels = []
metrics = {}

for subdir, _, files in os.walk(input_dir):
    folder_data = []
    folder_labels = []
    
    for file in files:
        if file.endswith('.html') or file.endswith('.htm'):
            file_path = os.path.join(subdir, file)
            try:
                extracted_data = extract_text_from_html(file_path)
                if not extracted_data:
                    print(f"No data extracted from file {file_path}")
                    continue
                for table in extracted_data:
                    category = categorize_table(table)
                    cleaned_text = clean_text(' '.join([' '.join(row) for row in table]))
                    folder_data.append(cleaned_text)  # Clean and combine rows for model input
                    folder_labels.append(category)
                # Save to CSV (optional)
                df = pd.DataFrame(extracted_data)
                relative_path = os.path.relpath(subdir, input_dir)
                output_subdir = os.path.join(output_base_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_file_path = os.path.join(output_subdir, f'{os.path.splitext(file)[0]}.csv')
                df.to_csv(output_file_path, index=False)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    if folder_data:
        # Convert folder data to DataFrame
        combined_data = pd.DataFrame({'text': folder_data, 'category': folder_labels})

        # Feature extraction with Word2Vec
        tokenized_data = [text.split() for text in combined_data['text']]
        word2vec_model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
        
        def get_sentence_vector(sentence, model):
            words = sentence.split()
            vector = np.mean([model.wv[word] for word in words if word in model.wv] or [np.zeros(model.vector_size)], axis=0)
            return vector
        
        X = np.array([get_sentence_vector(text, word2vec_model) for text in combined_data['text']])
        y = combined_data['category']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Support Vector Machine Classifier
        clf = SVC()
        clf.fit(X_train, y_train)

        # Predict on the test set
        y_pred = clf.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        folder_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        metrics[os.path.basename(subdir)] = folder_metrics

        print(f"Metrics for folder {os.path.basename(subdir)}:")
        print(f"  Model Accuracy: {accuracy}")
        print(f"  Model Precision: {precision}")
        print(f"  Model Recall: {recall}")
        print(f"  Model F1 Score: {f1}")

if not all_data:
    print("No data extracted from any HTML files.")
else:
    # Save the model to a pickle file
    model_path = os.path.join(output_base_dir, 'svm_model.pkl')
    with open(model_path, 'wb') as model_file:
        pickle.dump(clf, model_file)
    
    print(f"Model saved to {model_path}")
