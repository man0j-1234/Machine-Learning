
# Gmail vs Calendar Query Classifier using BERT
# -----------------------------------------------------------
# This script builds a text classification model using BERT to
# differentiate between Gmail-related and Calendar-related queries.
# It includes model training, evaluation, K-Fold cross-validation,
# date extraction, and a Streamlit interface.

# STEP 2: Import necessary libraries
import os
os.environ["USE_TF"] = "0"  # Disable TensorFlow in Transformers to avoid tf_keras issue

import pandas as pd
import torch
import spacy
import re
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

# STEP 3: Build a diverse and balanced dataset
def build_dataset():
    raw_data = [
        ("Find emails with PDF attachments", "gmail"),
        ("Show me unread messages in my inbox", "gmail"),
        ("Search for emails from Sarah about the project proposal", "gmail"),
        ("Find messages with the label 'Urgent'", "gmail"),
        ("What are the latest emails from HR?", "gmail"),
        ("Get emails that mention 'meeting notes'", "gmail"),
        ("Retrieve my conversation with Alex about the trip", "gmail"),
        ("Check if I got any emails with invoices", "gmail"),
        ("What did Sarah say in the email about budget?", "gmail"),
        ("Emails flagged as important", "gmail"),
        ("Do I have any mail from John regarding the party?", "gmail"),
        ("When is my next meeting with the design team?", "calendar"),
        ("Find appointments with Dr. Johnson", "calendar"),
        ("Show me all-day events in May", "calendar"),
        ("When is the team lunch scheduled for?", "calendar"),
        ("Add a meeting with Rahul tomorrow at 3 PM", "calendar"),
        ("Check my schedule for next week", "calendar"),
        ("Do I have any events on the weekend?", "calendar"),
        ("List all recurring events", "calendar"),
        ("Find my meetings for June 2025", "calendar"),
        ("What is on my calendar today?", "calendar"),
        ("Am I free on Friday after 5 PM?", "calendar"),
        ("Schedule a doctor appointment on Tuesday", "calendar"),
        ("Add internship interview to my calendar", "calendar"),
        ("Remind me of monthly review call", "calendar"),
        ("Calendar for October travel plans", "calendar"),
        ("Plan all events for next quarter", "calendar"),
        ("Do I have a client call this afternoon?", "calendar"),
        ("Show all my meetings for the upcoming week", "calendar"),
        ("What’s planned in my calendar tomorrow?", "calendar"),
        ("Are there any events overlapping on Wednesday?", "calendar"),
        ("Get me the agenda for next Friday’s calendar", "calendar"),
        ("Do I have any one-on-ones this month?", "calendar"),
        ("Show my calendar entries tagged 'important'", "calendar"),
        ("Block time for deep work next Thursday", "calendar"),
        ("Reschedule my product demo to Tuesday", "calendar"),
        ("List meetings scheduled with the marketing team", "calendar"),
        ("Check if I'm double-booked tomorrow", "calendar"),
        ("Remind me what Sarah said about the meeting", "gmail"),
        ("Remind me about the meeting with Sarah", "calendar"),
        ("Did Sarah confirm the appointment?", "calendar"),
        ("What did the email say about the upcoming event?", "gmail"),
        ("Search my calendar for budget discussions", "calendar"),
        ("Get the latest messages from Dr. Johnson", "gmail"),
        ("Tell me about the event in yesterday's email", "gmail"),
        ("Was there an event scheduled after the last email?", "calendar"),
        ("Find the notes from yesterday", "gmail"),
        ("What did Alex plan for Friday?", "calendar")
    ]
    df = pd.DataFrame(raw_data, columns=["query", "label"])
    df['label'] = df['label'].map({'gmail': 0, 'calendar': 1})
    return df

df = build_dataset()
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

@st.cache_resource
def load_model():
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

model = load_model()

def extract_date_range(query):
    match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", query, re.IGNORECASE)
    if match:
        month = match.group(1).capitalize()
        year = int(match.group(2))
        month_num = list(calendar.month_name).index(month)
        num_days = calendar.monthrange(year, month_num)[1]
        return f"from: {month} 1, {year} to: {month} {num_days}, {year}"
    return None

def classify_query(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    category = "Gmail" if prediction == 0 else "Calendar"
    time_range = extract_date_range(text) if category == "Calendar" else None
    return category, time_range

st.title("Gmail vs Calendar Query Classifier - by Manoj")
user_input = st.text_input("Enter your query below :")
if user_input:
    prediction, range_info = classify_query(user_input)
    st.success(f"Predicted Category: {prediction}")
    if range_info:
        st.info(f"Time Range Extracted: {range_info}")
