import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample email dataset (text and labels)
emails = [
    "buy now, special offer - 50% off!",
    "meeting tomorrow at 2pm",
    "urgent: please review the document",
    "click to claim your prize",
    "reminder: submit your report"
]

labels = ['spam', 'not spam', 'spam', 'spam', 'not spam']
# Step 1: Tokenization and Lowercasing
# This step breaks down each email into individual words and converts them to lowercase.
# It's often the first step in text preprocessing.

# Corrected: Apply lower() to each individual email
emails = [email.lower() for email in emails]

# Step 2: Count Vectorization
# Create a CountVectorizer to convert the list of emails into a bag-of-words representation.
vectorizer = CountVectorizer()

# Fit the vectorizer to the emails and transform them into a bag-of-words matrix
X = vectorizer.fit_transform(emails)

# The resulting X is the bag-of-words representation of the text data
# Each row corresponds to a document (email), and each column corresponds to a word in the vocabulary.

# Now, you can inspect the vocabulary (unique words/terms)
vocabulary = vectorizer.get_feature_names_out()
print("Vocabulary (Unique Words/Terms):")
print(vocabulary)

# You can also access the bag-of-words matrix X
print("\nBag-of-Words Matrix (X):")
print(X.toarray())

# Create a bag-of-words representation of the emails
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
print(X)

# Create a Naive Bayes classifier
classifier = MultinomialNB()
print(classifier)

# Train the classifier on the data
classifier.fit(X, labels)

# Now, let's classify a new email
new_email = ["congratulations! you've won a gift voucher"]
new_email_vectorized = vectorizer.transform(new_email)
predicted_label = classifier.predict(new_email_vectorized)

print("Predicted Label for the New Email:", predicted_label[0])