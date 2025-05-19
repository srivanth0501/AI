# -*- coding: utf-8 -*-
"""
# **Google Devices Q&A Chatbot with Intent Classification**
"""

!pip install gradio

!python -m spacy download en_core_web_md

import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import gradio as gr
import numpy as np
# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('punkt')

# Load the dataset
df = pd.read_excel('/content/All About Google Devices.xlsx')
df.head()

"""
**WordNetLemmatizer:**

The purpose of lemmatization is to reduce words to their base or root form. For example, "running" becomes "run".

Usage: Applied to each word in the text to maintain uniformity and increase text processing efficiency.

**preprocess_text:**

Purpose: Converts text to lowercase before further processing.
Removing numbers and punctuation.
Tokenizing the text into words.
Lemmatizing the words.

Replace terms with synonyms (if they exist in the preset dictionary).
Usage: Cleans and standardises the supplied text.

**get_wordnet_pos:**

Purpose: Converts part-of-speech tags to WordNet POS tags, which are required for correct lemmatization. This guarantees that words are lemmatized according to their proper grammatical role (noun, verb, adjective, adverb).

Usage: Helps provide the right foundation form of words understanding their context.

**generate_patterns:**

Purpose: Generates regular expression patterns that match device names and categories in text. This enables the dynamic extraction of entities based on the dataset.

Usage: Creates patterns to recognize device names and categories referenced in the input text.

**extract_entities:**

The purpose is to identify items in the text using developed patterns. It searches for matches and classifies them based on whether they belong to devices or categories.

Usage: Using the defined patterns, it extracts pertinent items from input text."""

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define static synonyms dictionary
synonyms = {
    "cellphone": "phone",
    "smartphone": "phone",
    "laptop": "computer",
    "notebook": "computer",
    "watch": "device",
    "charging": "charging",
    "battery": "charging",
    "power": "charging",
    # Add more synonyms here
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in words]
    synonym_replaced_words = [synonyms.get(w, w) for w in lemmatized_words]
    return " ".join(synonym_replaced_words)

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def generate_patterns(df):
    devices = df['Device'].unique()
    categories = df['Category'].unique()
    patterns = {
        'device': r'\b' + '|'.join(map(re.escape, devices)) + r'\b',
        'category': r'\b' + '|'.join(map(re.escape, categories)) + r'\b'
    }
    return patterns

def extract_entities(text, patterns):
    entities = []
    for entity_type, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append((match, entity_type))
    return entities

# Generate patterns dynamically based on the dataset
patterns = generate_patterns(df)

# Apply preprocessing
df['Processed_Question'] = df['Question'].apply(preprocess_text)

# Fit label encoders
device_encoder = LabelEncoder().fit(df['Device'])
category_encoder = LabelEncoder().fit(df['Category'])

df['Encoded_Device'] = device_encoder.transform(df['Device'])
df['Encoded_Category'] = category_encoder.transform(df['Category'])

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Processed_Question'])

# Train intent classifier
X_train, X_test, y_train, y_test = train_test_split(X, df['Category'], test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Test intent classifier accuracy
y_pred = clf.predict(X_test)
print(f'Intent Classifier Accuracy: {accuracy_score(y_test, y_pred)}')

def classify_intent(question):
    question_vec = vectorizer.transform([question])
    category = clf.predict(question_vec)[0]
    return category

"""**Text preprocessing:**

Lowercasing: Makes all characters lowercase to ensure consistency.

Removing Numbers and Punctuation: Regular expressions are used to remove numbers and punctuation from text.
Tokenization: divides the text into distinct words.

Lemmatization: With the WordNet Lemmatizer, words are reduced to their base or root form.

Synonym Replacement: To standardize terminology, specified synonyms are used instead of words.

POS Tagging: Determines each word's part of speech so that the right lemmatization technique may be used.

**Pattern Generation and Entity Extraction:**

Pattern Generation: Generates regular expression patterns dynamically using unique devices and categories from the dataset.
Entity Extraction:
Identifies and extracts entities from text using the generated patterns.
"""

def get_answer(question):
    preprocessed_question = preprocess_text(question)
    question_vec = vectorizer.transform([preprocessed_question])
    similarity = cosine_similarity(question_vec, X)
    index = np.argmax(similarity)
    best_match = df.iloc[index]
    device = device_encoder.inverse_transform([best_match['Encoded_Device']])[0]
    category = classify_intent(preprocessed_question)

    # Extract entities from the question using dynamically generated patterns
    entities = extract_entities(question, patterns)

    return f"This question is most likely related to the device '{device}' and falls under the category '{category}'. Discovered entities: {entities}"

"""# **Chatbot**"""

def chatbot(input_text, chat_history):
    # Initialize chat_history if None
    if chat_history is None:
        chat_history = []

    # Get the response based on the input
    response = get_answer(input_text)

    # Append the new interaction to the history
    chat_history.append((input_text, response))

    # Return updated history and state
    return chat_history, chat_history

# Create the Gradio interface
interface = gr.Interface(
    fn=chatbot,
    inputs=[gr.Textbox(label="You", placeholder="Type your question here..."),
            gr.State()],
    outputs=[gr.Chatbot(label="Chatbot"),
             gr.State()],
    theme="default"
)

# Launch the interface
interface.launch(debug=True)