## Gmail vs Calendar Query Classifier

This project is a smart text classifier that predicts whether a user query is related to **Gmail** or **Google Calendar** using a transformer-based model (RoBERTa). It also includes a bonus feature to extract time ranges from calendar queries and a clean Streamlit interface for real-time testing.

---

## Project Overview

Many user queries are vague and context-dependent. For example:
- "Remind me about the meeting with Sarah" → Calendar  
- "Remind me what Sarah said about the meeting" → Gmail  

This model understands such subtle differences using contextual embeddings from RoBERTa.

---

## Features

- Gmail vs Calendar classification using RoBERTa
- Custom dataset with balanced queries
- K-Fold Stratified Validation for evaluation
- Class-weighted loss to handle imbalance
- Regex-based time range extraction
- Streamlit app for real-time testing

---

## Files Included

| File                        | Description                                      |
|----------------------------|--------------------------------------------------|
| `ML_Assignment.ipynb`      | Full training, evaluation, and K-Fold loop       |
| `ML_Assignment.py`         | Final inference script for Streamlit app         | 

---

## Dataset

- Custom-labeled dataset with **Gmail**, **Calendar**, and **Ambiguous** queries
- Balanced across both classes
- Example entries:

| Query                                              | Label     |
|---------------------------------------------------|-----------|
| Find emails with PDF attachments                  | Gmail     |
| When is my next meeting with the design team?     | Calendar  |
| Remind me what Sarah said about the meeting       | Gmail     |

---

## Model Evaluation

- Used 5-Fold Stratified Cross Validation
- Handled class imbalance using `compute_class_weight`
- Printed and plotted:
  - Loss curves
  - Accuracy and F1 scores
  - Confusion matrix

---

## Streamlit Interface

Launch the app with:
```bash
streamlit run ML_Assignment.py
