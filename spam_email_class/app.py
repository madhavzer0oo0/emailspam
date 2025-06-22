import streamlit as st
import pickle

import string
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
    'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'can', 'will', 'just', 'don', 'should', 'now'
}
def transform(text):
    text = text.lower()
    words = text.split()
    y = []
    
    for word in words:
        # Remove non-alphanumeric characters from word
        cleaned = ''.join(char for char in word if char.isalnum())
        
        # Only keep if result is not empty
        if cleaned:
            y.append(cleaned)
    text=y[:]
    y.clear()
    for word in text:
        if word not in stopwords and word not in string.punctuation:
            y.append(word)

    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

cv=pickle.load(open('cv.pkl','rb'))
tf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email/sms Spam classifier')

sms=st.text_input("enter a message")


if st.button('Predict'):
    transform_sms=transform(sms)
    cv_input=cv.transform([transform_sms])
    vector_input=tf.transform(cv_input)

    output=model.predict(vector_input)[0] 

    if output==1:
       st.header("Spam")
    else:
        st.header("Not spam")


