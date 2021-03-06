# Importing essential libraries
import pandas as pd
import pickle

df = pd.read_csv('train.csv')
df2 = df.drop(['Title'],axis = 'columns')


import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

STOP_WORDS = stopwords.words('english')

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

lemmatizer = WordNetLemmatizer()

corpus = []
for i in range(0,len(df2)):
    review = re.sub('[^a-z,A-Z,0-9,%₹$€’′]',' ',df2['Description'][i])
    # Lowercasing all strings
    review = review.lower()
    #Expanding contractions
    # Replace some of the imp symbols texts numbers into meaningful words for better understanding
    review = review.replace('%',' percent').replace('₹',' rupee').replace('$',' dollar').replace('€',' euro')\
                                .replace(',000,000','m').replace('000','k').replace('′',"'").replace("’","'")\
                                .replace("won't","will not").replace("can't",'can not').replace("shouldn't","should not")\
                                .replace("what's",'"what is"').replace("that's",'that is').replace("he's","he is")\
                                .replace("she's","she is").replace("it's","it is").replace("'ve"," have").replace("'re"," are")\
                                .replace("'ll"," will").replace("i'm","i am").replace("n't", " not")
    # Remove Urls
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    review = re.sub(url_pattern,'',review)
    # Remove Html tags
    html_remove = re.compile('<.*?>')
    review = re.sub(html_remove, '', review)

    # Tokenizing
    tokens = re.findall("[\w']+", review)
    review = tokens
    # Remove Punchuation
    new_lst=[]
    for i in review:
        for j in  string.punctuation:
            i=i.replace(j,'')
        new_lst.append(i)
    review = new_lst
    # Remove Digits
    nodig_lst=[]
    new_lists=[]
    for a in review:
        for b in string.digits:
            a=a.replace(b,'')
        nodig_lst.append(a)
    for a in  nodig_lst:
        if  a!='':
            new_lists.append(a)
    review = new_lists

    # WordNet lemmatization
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_tf = TfidfVectorizer(tokenizer=None,stop_words=None,max_df=0.75,max_features=1000,lowercase=False,ngram_range=(1,2))
train_vectors = vectorizer_tf.fit_transform(corpus)

train_df = pd.DataFrame(train_vectors.toarray(),columns=vectorizer_tf.get_feature_names())
train_df = pd.concat([train_df,df2['Class Index'].reset_index(drop = True)],axis = 1)

with open('model_pickle_tf_idf','wb') as file:
    pickle.dump(vectorizer_tf,file)

from sklearn.linear_model import LogisticRegression

X = train_df.drop(['Class Index'],axis = 'columns')
y = train_df['Class Index']

#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
#sklearn.linear_model.LogisticRegression
lr_model = LogisticRegression(penalty='l2',solver= 'newton-cg',multi_class= 'multinomial')
# L2 = L2 regularization helps to overcome (overfitting vs underfitting)
# solver = newton-cg ... For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
lr_model.fit(X,y)
pred = lr_model.predict(X)


with open('model_pickle_logistic','wb') as file:
    pickle.dump(lr_model,file)



