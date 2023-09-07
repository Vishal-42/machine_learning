import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Preprocessing
stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation)


def preprocess_text(text):
    text = str(text)
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords and token not in punctuation]
    return ' '.join(tokens)


def extract_opinion_terms(text):
    df = pd.read_csv('dataset.csv', nrows=100)
    # Convert the 'Sentences' column into a list of strings
    sentences = df['Sentences'].tolist()
    df['opinion term'] = df['opinion term'].fillna("Other")
    # Convert the target variables into a list of strings
    opinion_terms = df[['Opinion Term', 'opinion term']].astype(str)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(sentences, opinion_terms, test_size=0.2, random_state=42)

    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Vectorize the training and testing data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    text = vectorizer.transform(text)
    # Create the Random Forest Classifier
    clf = RandomForestClassifier()

    # Train the model
    clf.fit(X_train_vec, y_train)
    opinion_term = clf.predict(text)

    return opinion_term


# Opinion Category Classification
def train_category_classifier():
    # Load the dataset for opinion category classification
    df = pd.read_csv('dataset.csv')

    # Preprocess the text data
    df['preprocessed_text'] = df['Sentences'].apply(preprocess_text)

    # Get unique aspect categories
    aspect_categories = ['Aspect Category', 'aspect  category', 'aspect category ', 'aspect  category ']

    # Fill NaN values with a default category
    df[aspect_categories] = df[aspect_categories].fillna('Other')

    # Split the df into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['preprocessed_text'], df[aspect_categories],
                                                        test_size=0.2, random_state=42)

    # Vectorize the preprocessed text data
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train a Support Vector Machine (SVM) classifier for each aspect category
    classifiers = {}
    for aspect_category in aspect_categories:
        classifier = SVC()
        classifier.fit(X_train_vec, y_train[aspect_category])
        classifiers[aspect_category] = classifier

    return vectorizer, classifiers


# Polarity Assignment
def train_polarity_classifier():
    # Load the dataset for polarity assignment
    df = pd.read_csv('dataset.csv')

    # Preprocess the text data
    df['preprocessed_text'] = df['Sentences'].apply(preprocess_text)

    # Get unique aspect polarities
    aspect_polarities = ['Aspect Polarity ', 'Aspect polarity', 'Aspect polarity ', 'aspect polarity ']

    # Fill NaN values with a default polarity
    df[aspect_polarities] = df[aspect_polarities].fillna('Neutral')

    # Split the df into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['preprocessed_text'], df[aspect_polarities],
                                                        test_size=0.2, random_state=42)

    # Vectorize the preprocessed text data
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train a Support Vector Machine (SVM) classifier for each aspect polarity
    classifiers = {}
    for aspect_polarity in aspect_polarities:
        classifier = SVC()
        classifier.fit(X_train_vec, y_train[aspect_polarity])
        classifiers[aspect_polarity] = classifier

    return vectorizer, classifiers


# Predictions
def predict_opinion_terms_categories_polarity(text):
    sub_sentences = [sentence.strip() for sentence in text.split('.') if sentence.strip() != '']
    opinion_terms_list = []
    categories_list = []
    polarities_list = []

    # Opinion Term Extraction
    opinion_terms = extract_opinion_terms(sub_sentences)
    opinion_terms_list.append(opinion_terms)
    for sub_sentence in sub_sentences:
        # Opinion Category Classification
        vectorizer_category, classifiers_category = train_category_classifier()
        preprocessed_text = preprocess_text(sub_sentence)
        X_vec_category = vectorizer_category.transform([preprocessed_text])

        categories = []
        for aspect_category, classifier in classifiers_category.items():
            category = classifier.predict(X_vec_category)[0]
            if category != 'Other':
                categories.append(category.lower())

        categories_list.append(categories)

        # Polarity Assignment
        vectorizer_polarity, classifiers_polarity = train_polarity_classifier()
        X_vec_polarity = vectorizer_polarity.transform([preprocessed_text])

        polarities = []
        for aspect_polarity, classifier in classifiers_polarity.items():
            polarity = classifier.predict(X_vec_polarity)[0]
            if polarity != 'Neutral':
                polarities.append(polarity)

        polarities_list.append(polarities)

    aspect_categories = []
    for aspect_category in categories_list:
        for ac in aspect_category:
            if ac != "" and ac not in aspect_categories:
                aspect_categories.append(ac)

    opinion_terms = []
    for opinion_term in opinion_terms_list:
        for ot in opinion_term:
            opinion_terms.append(ot[0])

    aspect_polarities = []
    for aspect_polarity in polarities_list:
        for ap in aspect_polarity:
            aspect_polarities.append(ap)

    return opinion_terms, aspect_categories, aspect_polarities


if __name__ == '__main__':
    sentence = "Absolutely terrible professionals .. Twice had bad experience with them.. They did not do anything about it.. The professionals are rude.. Pricing is way too high.. They charge for hours where the professional has gone to the shop to buy materials also.."
    opinion_terms, aspect_categories, aspect_polarities = predict_opinion_terms_categories_polarity(sentence)
    print("Opinion Terms:", opinion_terms)
    print("Aspect Categories:", aspect_categories)
    print("Aspect Polarities:", aspect_polarities)