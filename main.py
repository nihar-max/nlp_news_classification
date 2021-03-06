#pip install fastapi uvicorn

# 1. Library imports
import uvicorn ##ASGI
from fastapi import FastAPI
import streamlit as st
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pydantic import BaseModel
import string
import pickle
from bs4 import BeautifulSoup
import numpy as np
app = FastAPI()

class News(BaseModel):
    news: str


# Load the Tf-Idf model and Logistic regression object from disk
pickle_in = open("model_pickle_tf_idf","rb")
tf_idf_model=pickle.load(pickle_in)
pickle_ml_in = open("model_pickle_logistic","rb")
ml_model=pickle.load(pickle_ml_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome': f'{name}'}


@app.post('/predict')
def predict_news(data:News):
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    news = data.news
    # Lowering the srings
    lower_str = news.lower()
    # Expanding Contradictions
    expanding_contradictions = lower_str.replace('%',' percent').replace('₹',' rupee').replace('$',' dollar').replace('€',' euro')\
                                .replace(',000,000','m').replace('000','k').replace('′',"'").replace("’","'")\
                                .replace("won't","will not").replace("can't",'can not').replace("shouldn't","should not")\
                                .replace("what's",'"what is"').replace("that's",'that is').replace("he's","he is")\
                                .replace("she's","she is").replace("it's","it is").replace("'ve"," have").replace("'re"," are")\
                                .replace("'ll"," will").replace("i'm","i am").replace("n't", " not")\
                                .replace(" 4 "," four ").replace(" 3 "," three ").replace(" 2 "," two ")\
                                .replace(" 1 "," one ").replace(" 0 "," zero ").replace(" 5 "," five ")\
                                .replace(" 8 "," eight ").replace(" 7 "," sevem ").replace(" 6 "," six ").replace(" 9 "," nine ")
    expanding_contradictions = re.sub(r"([0-9]+)000000", r"\1m", expanding_contradictions)
    expanding_contradictions = re.sub(r"([0-9]+)000", r"\1k", expanding_contradictions)

    pattern = re.compile('\W')
    if type(expanding_contradictions) == type(''):
        only_words = re.sub(pattern, ' ', expanding_contradictions)



    # Removing urls
    url = re.compile(r'https?://\S+|www\.\S+')
    clean_url = re.sub(url, ' ', only_words)

    # Remove Html tags
    html_pattern = re.compile('<.*?>')
    clean_html = re.sub(html_pattern, ' ', clean_url)

    # Removing Punctuations
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punc = ""
    for char in clean_html:
        if char not in punctuations:
            no_punc = no_punc + char

    # Removing Numbers
    numbers = string.digits
    no_digit = ""
    for char in no_punc:
        if char not in numbers:
            no_digit = no_digit + char

     # Creating Tokens
    split_words = no_digit.split()
    STOP_WORDS = stopwords.words('english')

    # Stopwords removal as well as other ExtraWords removal
    stopwords = set(STOP_WORDS)
    stopwords.add("said")
    stopwords.add("br")
    stopwords.remove("not")
    stopwords.remove("no")
    stopwords.add(" ")
    stopwords.add("href")
    stopwords.add("html")
    stopwords.add("www")
    stopwords.add("quot")
    stopwords.add("gt")
    stopwords.add("lt")
    stopwords.add("ii")
    stopwords.add("iii")
    stopwords.add("ie")
    stopwords.add("com")
    text_preprocessing = [lemmatizer.lemmatize(word) for word in split_words if not word in stopwords]
    tf_idf_model_training = tf_idf_model.transform(text_preprocessing)
    prediction = ml_model.predict(tf_idf_model_training)
    if prediction[0] == 1:
        prediction = 'World News'
    elif prediction[0] == 2:
        prediction = 'Sports News'
    elif prediction[0] == 3:
        prediction = 'Business News'
    else:
        prediction = 'Science & Tech News'
    return {
        'prediction': prediction
    }

    

    # tf_idf = tf_idf_model.transform(news)
    # prediction     = ml_model.predict(tf_idf)
    # if prediction == 1:
    #     prediction == 'world'
    # elif prediction == 2:
    #     prediction == 'sports'
    # elif prediction == 3:
    #     prediction == 'buisness'
    # else:
    #     prediction == 'science'
    # return {
    #     'prediction' : prediction
    # }


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
   uvicorn.run(app, host='127.0.0.1', port=8080)



#uvicorn main:app --reload