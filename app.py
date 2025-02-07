import streamlit as st
import numpy as np
import tensorflow
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,Embedding
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

model=load_model('model.h5')
word_index=imdb.get_word_index()

st.title('IMDB Movie Review')
text=st.text_input('Movie Review')
submit=st.button('Submit')
   
  

# Function to preprocess the user input
def preprocess_review_fn(text, word_index, max_len=1000):
    words = text.lower().split()  # Tokenize the input by splitting on spaces
    encoded_review = [word_index.get(word, 2) for word in words]  # 2 is for unknown words (OOV token)
    padded_review = pad_sequences([encoded_review], maxlen=max_len)  # Ensure input is of the correct length
    return padded_review

# Prediction function
def prediction_fn(review, model, word_index):
    review_data = preprocess_review_fn(review, word_index)
    prediction = model.predict(review_data)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]
if submit:
    if text:

        # Assuming the word_index and reversed_word_index are predefined and the model is loaded
    
        sentiment, score = prediction_fn(text, model, word_index)

        st.write('Sentiment:', sentiment)
        st.write('Score:', score)
    else:
        st.write('Please enter a review')


