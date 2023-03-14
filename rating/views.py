from django.shortcuts import render
from django.http import HttpResponse


from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import pickle
import os
import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
import numpy as np

model = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'files/mod_klass5.h5'),compile=False)
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy']) 

model_bin = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'files/mod_klass_bin4.h5'),compile=False)
model_bin.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) 


with open(os.path.join(os.path.dirname(__file__), 'files/scaler.pkl'), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(os.path.dirname(__file__), 'files/count.pkl'), "rb") as f:
    count = pickle.load(f)
    
with open(os.path.join(os.path.dirname(__file__), 'files/select_class.pkl'), "rb") as f:
    select_class = pickle.load(f)

lemmatize = nltk.WordNetLemmatizer()

def text_for_pred(text):
    text=re.sub("[^a-zA-Z]"," ",text)
    text = nltk.word_tokenize(text,language = "english")
    text = [lemmatize.lemmatize(word) for word in text]
    # лемматирзируем слова
    # соединяем слова
    text = " ".join(text)
    text=pd.DataFrame({'as':[text],})
    matrix = count.transform(text['as']).toarray()
    matrix=select_class.transform(matrix)
    matrix = scaler.transform(matrix)

    return matrix






def index(request):
    return render(request, "index.html")
 
def postuser(request):
    # получаем из данных запроса POST отправленные через форму данные
    name = request.POST.get("name", "Undefined")
    


    predictions = model.predict(text_for_pred(name))
    rating=np.argmax(predictions)

    predictions_bin = model_bin.predict(text_for_pred(name))

    pre=''
    for i in predictions_bin:
        if i<0.5 and rating<5:
            pre='Негативная'
        else:
            pre='Позитивная'

    return HttpResponse(f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Определение вероятного рейтинга по рецензии</title>
</head>
<body style="margin:40px; background:MintCream;">
<h1 style="text-align:center;">Результаты оценки вашей рецензии<h1> <br>
<h2>Рецензия:<h2> <br> 
<p>{name}</p> <br> 
<p style="font-size:35px; "> Предполагаемый рейтинг фильма : {rating} </p> <br>
<p style="font-size:35px; "> Предполагаемый характер рецензии : {pre}</p>
</body>
</html>
''')