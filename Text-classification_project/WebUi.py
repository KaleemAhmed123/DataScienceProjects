import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

psmr = PorterStemmer()

# This function performs all 5 steps


def text_transformation(text):
    text = text.lower()                  # Lowering chars
    text = nltk.word_tokenize(text)      # tokenizing in word

    processed_text = []
    for i in text:                       # Removing special chars
        if i.isalnum():
            processed_text.append(i)

    text = processed_text[:]            # Cloning cause copy will empatise
    processed_text.clear()

    for i in text:                      # 4th removing stopwords and punctuations
        if i not in stopwords.words("english") and i not in string.punctuation:
            processed_text.append(i)

    text = processed_text[:]
    processed_text.clear()

    for i in text:
        processed_text.append(psmr.stem(i))   # Stemming step

    return " ".join(processed_text)


cv = pickle.load(open('vectorizer.pkl', 'rb'))  # reading
model = pickle.load(open('modelFinal.pkl', 'rb'))

st.title("SMS/Email Spam Classifier")

input_sms = st.text_area("Type the message here")

if st.button('Predict Button'):

    # Data preprocessing bnlp step

    preprocessed_input = text_transformation(input_sms)

    # Followed by CouuntVectorizer step

    cv_vectorize_input = cv.transform([preprocessed_input])

    #  prediction step

    result = model.predict(cv_vectorize_input)[0]

    #  Displaying the result to UI

    if result == 1:
        st.header("Spam message")
    else:
        st.header("Not a Spam message")

    # Thanks streamlit devs
