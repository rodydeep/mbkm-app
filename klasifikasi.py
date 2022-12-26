# -*- coding: utf-8 -*-
"""klasifikasi_NB.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_rMktey4rSZGrXimtNoGIENu8hSQiS_4
"""

import pandas as pd
import pandas as pd
import numpy as np
import string
import re 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from textblob import TextBlob

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

"""# Bagian Baru"""

dataset = pd.read_excel("data_label.xlsx")
dataset.head()

dataset = dataset[['Content','Label']]
dataset = dataset.rename(columns={'Content':'text','Label':'label'})
dataset.head()

dataset['label'].value_counts()

dataset['text'] = dataset['text'].astype(str)

vec = CountVectorizer().fit(dataset['text'])
vec_transform = vec.transform(dataset['text'])
print(vec_transform)

x = vec_transform.toarray()
y = dataset['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=21)

print("data Latih x : ", x_train.shape)
print("data Uji x : ", x_test.shape)
print("data Latih y : ", y_train.shape)
print("data Uji y : ", y_test.shape)

metodeNB = MultinomialNB().fit(x_train, y_train)

predictNB = metodeNB.predict(x_test)

print(confusion_matrix(y_test, predictNB))
print("\n")
print(classification_report(y_test, predictNB))

from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
style.use('classic')
cm = confusion_matrix(y_test, predictNB, labels=metodeNB.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=metodeNB.classes_)
disp.plot()

data_pred = pd.read_excel("data uji.xlsx")
data_pred.head()

data_pred = data_pred[['Content']]
data_pred = data_pred.rename(columns={'Content':'text'})
data_pred.head()

from scipy.sparse import data
data_pred['vector'] = data_pred['text'].astype(str)

vec = CountVectorizer().fit(data_pred['vector'])
vec_transform = vec.transform(data_pred['vector'])
print(vec_transform)
x_test = vec_transform.toarray()

data_pred.head()

data_pred['sentimen'] = metodeNB.predict(x_test)

data_pred.head()

data_pred['sentimen'].value_counts()

count = data_pred['sentimen'].value_counts()
plt.figure(figsize=(10, 6))


plt.bar(['Positive', 'Netral', 'Negative'], count, color=['royalblue','green', 'orange'])

plt.xlabel('Jenis Sentimen', size=14)
plt.ylabel('Frekuensi', size=14)
plt.show()

def casefolding(text):
  text = text.lower()         #merubah kalimat menjadi huruf kecil
  return text

def cleaning(text):
    text = text.replace('\\t',"").replace('\\n',"").replace('\\u',"").replace('\\',"")

    text = text.encode('ascii', 'replace').decode('ascii')

    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\s+)"," ",text).split())

    text = re.sub(r'http\S+', '',text)
    text = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ",text)
    text = re.sub(r'http\S+', '',text)
    text = text.translate(str.maketrans(" ", " ", string.punctuation))
    text = text.strip()
    text = re.sub(r"\b[a-z A-Z]\b", " ", text)
    text = re.sub("/s+", " ", text)
    text = re.sub(r"\b[a-z A-Z]\b", " ", text)
    
    return text

def preprocess_data(text):
  text = casefolding(text)
  text = cleaning (text)

  return text

!pip install streamlit

import streamlit as st

st.title("Aplikasi analisis sentiment data twitter")

# Load the dataset
df = st.sidebar.file_uploader("Upload a dataset in xlsx or csv format", type=["xlsx", "csv"])
if df:
    df = pd.read_excel(df) if df.name.endswith('xlsx') else pd.read_csv(df)
    st.sidebar.success("Upload data selesai")

if st.sidebar.checkbox("prediksi sentimen"):
  def preprocess_data(text):
    text = casefolding(text)
    text = cleaning (text)

    return text
  df["text"]= df["Content"].apply(preprocess_data)
st.sidebar.success("prepocessing selesai")

# klasifikasi model
if st.sidebar.checkbox("prediksi sentimen"):
    df['vector'] = df['text'].astype(str)

    vec = CountVectorizer().fit(df['vector'])
    vec_transform = vec.transform(df['vector'])

    x_test = vec_transform.toarray()

    df['sentimen'] = metodeNB.predict(x_test)
   
   
st.sidebar.success("Prediksi Sentimen Selesai")
st.sidebar.dataframe(df)


# Make predictions
if st.sidebar.checkbox("visualisasi"):
    count = df['sentimen'].value_counts()
    plt.figure(figsize=(10, 6))


    plt.bar(['Positive', 'Netral', 'Negative'], count, color=['royalblue','green', 'orange'])

    plt.xlabel('Jenis Sentimen', size=14)
    plt.ylabel('Frekuensi', size=14)
    plt.show()
st.sidebar.success("visualisasi Selesai")