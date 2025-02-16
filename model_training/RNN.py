import os
from bs4 import BeautifulSoup
import pandas as pd
import chardet
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, LSTM

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

        # Tokenize text data
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(combined_data['text'])
        X = tokenizer.texts_to_sequences(combined_data['text'])
        X = pad_sequences(X)

        # Convert labels to categorical
        y = pd.get_dummies(combined_data['category'])

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the RNN model
        model = Sequential()
        model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=X.shape[1]))
        model.add(SimpleRNN(units=64))
        model.add(Dense(units=y.shape[1], activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.2)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
        folder_metrics = {
            'loss': loss,
            'accuracy': accuracy
        }
        metrics[os.path.basename(subdir)] = folder_metrics

        print(f"Metrics for folder {os.path.basename(subdir)}:")
        print(f"  Model Loss: {loss}")
        print(f"  Model Accuracy: {accuracy}")


    # Save the model to a pickle file
model_path = os.path.join(output_base_dir, 'rnn_model.pkl')
model.save(model_path)
    
print(f"Model saved to {model_path}")
