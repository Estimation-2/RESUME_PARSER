# Create and Save clf.pkl, tfidf.pkl, and encoder.pkl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Sample Resume Dataset (Modify as needed)
data = {
    "Resume": [
        "Experienced data scientist skilled in Python, machine learning, and data analysis.",
        "Software developer experienced in Java, Spring Boot, and microservices.",
        "Network security engineer with expertise in firewalls, IDS/IPS, and cyber forensics.",
        "Health and fitness expert with certifications in personal training and nutrition."
    ],
    "Category": ["Data Scientist", "Software Developer", "Network Security Engineer", "Fitness Trainer"]
}

# Load data into DataFrame
df = pd.DataFrame(data)

# Text Preprocessing Function
def clean_text(text):
    import re
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Clean the Resume Text
df["Cleaned_Resume"] = df["Resume"].apply(clean_text)

# Train TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=100)
X = tfidf.fit_transform(df["Cleaned_Resume"])

# Encode Labels
le = LabelEncoder()
y = le.fit_transform(df["Category"])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Save Model Files
pickle.dump(model, open("clf.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("Model, TF-IDF, and Encoder files created successfully!")
