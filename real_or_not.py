import numpy as np
import pandas as pd
import nltk
import re
import string
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import linear_model, model_selection
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

train_data = pd.read_csv("train.csv", encoding="L1")
test_data = pd.read_csv("test.csv")

# one hot vector
count_vectorizer = CountVectorizer()

def  clean_text (text) :
# make text lowercase
    text = text.lower()
 # remove square brackets
    text = re.sub( '\[.*?\]' , '' , text)
 # remove links
    text = re.sub( 'https?://\S+|www\.\S+' , '' , text)
 # remove <>
    text = re.sub( '<.*?>+' , '' , text)
 # remove punctuation
    text = re.sub( '[%s]' % re.escape(string.punctuation),'' , text)
# remove \n
    text = re.sub( '\n' , '' , text)
# remove numbers
    text = re.sub( '\w*\d\w*' , '' , text)
#删除text列中的特殊字元
    text = re.sub('Û|û|Å|å|Â|â|Ã|ã|Ê|ê|È|è|ï|Ï|Ì|ì|Ó|ó|Ò|ò|ª|ñ���|_','',text)
    return text
train_data['text']=train_data['text'].apply(lambda x : clean_text(x))

#Tokenization
#print(train_data['text'].head())
train_data['text']=train_data['text'].apply(lambda x:word_tokenize(x))

#stop word
def remove_stopwords(words):
    text=[]
    for word in words:
        if word not in stopwords.words('english'):
            text.append(word)
    return text
train_data['text'] = train_data['text'].apply(lambda x: remove_stopwords(x))

#獲取詞性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

#還原型
def Lemmatizing(word):
    text=[]
    tagged_sent = pos_tag(word)
    wnl = WordNetLemmatizer()
    for tag in tagged_sent:
        wordnet_pos = (get_wordnet_pos(tag[1]) or wordnet.NOUN)
        text.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    return text
train_data['text']=train_data['text'].apply(lambda x : Lemmatizing(x))

# spelling corrected
def correct_spellings(words):
    spell = SpellChecker()
    corrected_text = []
    for word in words:
        misspelled_words = spell.unknown(word)
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return corrected_text

train_data['text']=train_data['text'].apply(lambda x : correct_spellings(x))

#final text
def final_text(words):
     return ' '.join(words)
train_data['text']=train_data['text'].apply(lambda x:final_text(x))
#print(train_data['text'].head())


X_test = count_vectorizer.transform(test_data["text"])
X_train = count_vectorizer.fit_transform(train_data["text"])
X_train_text = count_vectorizer.get_feature_names()
y_train = train_data["target"]
print(X_train.todense().shape)
#print(X_train.todense().shape)

model = [linear_model.SGDClassifier(loss='modified_huber', penalty='l1', alpha=1e-05, n_iter_no_change=5, random_state=42),
        linear_model.LogisticRegression(C=50,multi_class='ovr', penalty='l2', tol=0.1,solver='sag'),
        SVC(C=100, gamma=0.001, kernel='rbf', probability=True),
        MultinomialNB()]

for clf in model:
    scores = model_selection.cross_val_score(clf, X_train, y_train, cv=3, scoring="f1")
    print(clf.__class__.__name__, scores)
    y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    print(conf_mx)

clf.fit(X_train, y_train)
sample_submission = pd.read_csv("sample_submission.csv")
sample_submission["target"] = clf.predict(X_test)
sample_submission.to_csv("submission.csv", index=False)
