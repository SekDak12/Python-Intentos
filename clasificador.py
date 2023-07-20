#!/usr/bin/env python
# coding: utf-8

# In[16]:


##Clasificador de correos spam mediante sklearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('mail_data.csv', encoding='latin-1')



# In[4]:


data = df.where(pd.notnull(df), "")


# In[33]:


data.head()


# In[6]:


##Renombramos las categorias como variables binarias

data.loc[data['Category'] == 'spam', 'Category',] = 0 

data.loc[data['Category'] == 'ham', 'Category',] = 1


# In[7]:


X = data['Message']
Y = data['Category'] 


# In[8]:


#Se dividen los diferentes grupos tanto de entrenamiento como de testing, tomando gurpo de 80% de entrenamiento y 20% de testing

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 3)

#Se puede comprobar imprimiendo los .shape de cada uno de los grupos


# In[9]:


feature_extraction = TfidfVectorizer(min_df = 1, stop_words = "english", lowercase=True)

X_train_features= feature_extraction.fit_transform(X_train)
X_test_features= feature_extraction.transform(X_test)
 
Y_train = Y_train.astype("int")
Y_test = Y_test.astype("int")


# In[53]:


##Modelo pero esta ves con Arbol de decision
algoritmo = DecisionTreeClassifier() 
algoritmo1 = DecisionTreeClassifier(max_depth=4)
algoritmo2  = DecisionTreeClassifier(max_depth=4, min_samples_split=5)
algoritmo.fit(X_train_features, Y_train)
algoritmo1.fit(X_train_features, Y_train)
algoritmo2.fit(X_train_features, Y_train)
0.9183856502242153  #Para arbol de decision sin restricciones
0.9668161434977578  #Profundidad maxima del arbol de 4 
0.9174887892376682  #Con la restriccion de profundidad maxima igual a 4 y a minimo 5 elentos por hoja


# In[54]:


##Prediccion sobre el set de testing
prediccion_testing_arbol = algoritmo2.predict(X_test_features)
promedio_arbol = accuracy_score(Y_test, prediccion_testing_arbol)

print(promedio_arbol)


# In[51]:


##Prediccion sobre el set de entrenamiento
prediccion_train_arbol = algoritmo.predict(X_train_features)
promedio_arbol = accuracy_score(Y_train, prediccion_train_arbol)

print(promedio_arbol)


# In[52]:


##Modelo pero esta ves con Regresion logistica

model = LogisticRegression()
model.fit(X_train_features, Y_train)


# In[11]:


prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print("Acc on training data:" , accuracy_on_training_data)


# In[25]:


prediction_on_testing_data = model.predict(X_test_features)
accuracy_on_testing_data = accuracy_score(Y_test, prediction_on_testing_data)
print("Acc on testing data:" , accuracy_on_testing_data)


# In[38]:


#Pruebas con nuevos correos introducidos como cadenas de texto con regresion logistica
input_your_email=["Free entry in 2 a wkly comp to win FA Cup in may"]
input_data_features = feature_extraction.transform(input_your_email)
prediction = model.predict(input_data_features)
print(prediction)


# In[39]:


#Pruebas con nuevos correos introducidos como cadenas de texto con arbol de decision
input_your_email=["Free entry in 2 a wkly comp to win FA Cup in may"]
input_data_features = feature_extraction.transform(input_your_email)
prediction = algoritmo.predict(input_data_features)
print(prediction)


# In[ ]:





# In[ ]:




