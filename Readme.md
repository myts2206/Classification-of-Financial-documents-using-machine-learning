# Financial Data Classification Report

## Introduction
This report summarizes the approach, model selection, and results for a financial data classification task. The objective was to classify financial data extracted from HTML files into predefined categories using various machine learning and deep learning models.

## Approach
The approach to this task was divided into several key steps:

### Data Extraction and Preprocessing:
- HTML files were parsed to extract text data from tables.
- Extracted text was cleaned by removing stop words, punctuation, and applying lemmatization.

### Categorization:
- Tables were categorized into the following groups using regex patterns and keyword matching:
  - Income Statements
  - Balance Sheets
  - Cash Flows
  - Notes
  - Others

### Model Selection:
- Multiple models were trained and evaluated, including:
  - Recurrent Neural Network (RNN)
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
- Text data was tokenized and padded for input into the models.
- Labels were one-hot encoded for classification tasks.

### Training and Evaluation:
- Data was split into training and testing sets (80-20 split).
- Models were trained using the training set and evaluated on the testing set.
- Metrics used for evaluation included accuracy and loss.

## Model Performance
The following models were trained and evaluated on the dataset:
- **RNN:** 100% accuracy
- **Decision Tree:** 100% accuracy
- **Random Forest:** 100% accuracy
- **SVM:** 100% accuracy
- **XGBoost:** 98% accuracy

## Conclusion
The analysis demonstrated that the selected models perform exceptionally well in classifying financial data, with RNN, Decision Tree, Random Forest, and SVM achieving perfect accuracy, and XGBoost closely following with 98% accuracy.

The high accuracy rates indicate that the models can reliably classify financial documents into predefined categories. Future work could explore:
- Hyperparameter tuning
- Incorporating more diverse datasets
- Enhancing model robustness

---
