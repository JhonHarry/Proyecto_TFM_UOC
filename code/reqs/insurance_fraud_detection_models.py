#!/usr/bin/env python
# coding: utf-8

# <p style = "font-size : 50px; color : #532e1c ; font-family : 'Calibri'; text-align : center; background-color : #D3D3D3; border-radius: 5px 5px;"><strong>Deteccion de fraude en seguros de vehiculos</strong></p>

# # Machine learning aplicado al fraude en seguros de vehiculos - TFM 
# ## Jhon Harry Loaiza

# ### Summary

# El conjunto de datos objeto de análisis se llama  [*insurance_claims.csv*](https://www.kaggle.com/toramky/automobile-dataset), el cual se ha obtenido en Kaggle y está conformado por 40 columnas y 1000 filas, la descripcion de los campos del dataset es la siguiente:
# 
# La descripción de las variables del dataset es la siguiente:
# 
# * **months_as_customer**: tiempo en meses que el cliente ha estado como cliente de seguros de vehículos
# * **age**: edad del cliente
# * **policy_number**: numero de poliza de seguros de vehiculos
# * **policy_bind_date**: fecha de inicio de la poliza de seguros de vehiculos
# * **policy_state**: estado de la poliza de seguros de vehiculos
# * **policy_csl**: clasificacion de la poliza de seguros de vehiculos
# * **policy_deductable**: deducible de la poliza de seguros de vehiculos
# * **policy_annual_premium**: cuota anual de la poliza de seguros de vehiculos
# * **umbrella_limit**: limite de la poliza de seguros de vehiculos
# * **insured_zip**: codigo postal del cliente
# * **insured_sex**: genero del cliente de la poliza de seguros
# * **insured_education_level**: nivel educativo del cliente asegurado
# * **insured_occupation**: ocupacion del asegurado
# * **insured_hobbies**: hobbies del cliente de la poliza de seguros de vehiculos
# * **insured_relationship**: relacion del cliente de la poliza de seguros de vehiculos
# * **capital-gains**: capital ganado del cliente de la poliza de seguros de vehiculos
# * **capital-loss**: capital perdido del cliente de la poliza de seguros de vehiculos
# * **incident_date**: fecha de incidente del cliente de la poliza de seguros de vehiculos
# * **incident_type**: tipo de incidente del cliente de la poliza de seguros de vehiculos
# * **collision_type**: tipo de colision del cliente de la poliza de seguros de vehiculos
# * **incident_severity**: gravedad del incidente del cliente de la poliza de seguros de vehiculos
# * **incident_state**: estado del incidente del cliente de la poliza de seguros de vehiculos 
# * **incident_city**: ciudad del incidente del cliente de la poliza de seguros de vehiculos
# * **incident_location**: localizacion del incidente del cliente de la poliza de seguros de vehiculos
# * **incident_hour_of_the_day**: hora del dia del incidente del cliente de la poliza de seguros de vehiculos
# * **number_of_vehicles_involved**:  numero de vehiculos involucrados en el incidente del cliente de la poliza de seguros de vehiculos
# * **property_damage**: daño a la propiedad del cliente de la poliza de seguros de vehiculos
# * **bodily_injuries**: heridos del cliente de la poliza de seguros de vehiculos
# * **witnesses**: testigos del cliente de la poliza de seguros de vehiculos
# * **police_report_available**: reporte de la policia del cliente de la poliza de seguros de vehiculos
# * **total_claim_amount**: total de la reclamacion del cliente de la poliza de seguros de vehiculos
# * **injury_claim**: reclamacion de heridos del cliente de la poliza de seguros de vehiculos
# * **property_claim**: reclamacion de la propiedad del cliente de la poliza de seguros de vehiculos
# * **vehicle_claim**: reclamacion del vehiculo del cliente de la poliza de seguros de vehiculos
# * **auto_make**: marca del vehiculo del cliente de la poliza de seguros de vehiculos
# * **auto_model**: modelo del vehiculo del cliente de la poliza de seguros de vehiculos
# * **auto_year**: año del vehiculo del cliente de la poliza de seguros de vehiculos
# * **fraud_reported**: reporte de fraude del cliente de la poliza de seguros de vehiculos

# ## 0. Cargue de librerias

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import plotly.express as px
from collections import Counter
from boruta import BorutaPy

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier 
import xgboost as xgb
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.metrics import precision_score,recall_score,accuracy_score, jaccard_score, cohen_kappa_score, log_loss ,f1_score, roc_auc_score


import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Cargue de la base de datos y descripcion del dataset

# In[2]:


# cargue de datos
df = pd.read_csv(r'C:\Users\owner\Documents\Master UOC\Semestre V\TFM\Proyecto_TFM_UOC\dataset\insurance_claims.csv')
# Dimensión de los datos
print("Dimensiones del dataset:" + str(df.shape))
df.info()


# In[3]:


""" for column in df:
    print(column)
    print(sorted(df[column].unique()), '\n') """


# In[4]:


np.min(df['auto_year'])


# In[5]:


df.head()


# ## 2. Preparacion de datos

# Reemplazamos los valores con '?' observados en las variables categoricas como **police_report_available**. Crearemos una funcion que permite identificar los campos que posean datos perdidos. 
# Tambien existe una columna vacia de nombre **_c39** la cual puede ser descartada mediante esta funcion, al hacer un drop de las columnas que posen mas de un 60% de datos perdidos.

# In[6]:


def drop_missing(df):
    """_summary_ 
    Retorna las columnas que posean 
    menos del 60% de datos perdidos

    Args:
        df (DataFrame): _description_

    Returns:
        _Dataframe con columnas con menos del 60%
        de valores perdidos
    """

    threshold = len(df) * 0.6
    df.dropna(axis=1, thresh=threshold, inplace=True)
    return df


# In[7]:


# verificamos las columnas remanentes en el dataset
drop_missing(df)


# Identificamos los valores marcados como "?" usando un loop, para identificar si la columna contiene el simbolo entre sus valores

# In[8]:


def quotation_values(df):
    """retorna columnas y el conteo 
        con valores '?'

    Args:
        df (_type_): argumento de entrada es
        un dataframe

    Returns:
        dataframe con los campos y la suma de records
        con '?'
    """
    
    perdidosList = {}
    for col in list(df.columns):
        if (df[col]).dtype == object:
            quotation = np.sum(df[col] == '?')
            perdidosList[col] = quotation
    perdidos = pd.DataFrame.from_dict(perdidosList, orient = 'index')
    perdidos.columns =['Count_lost']
    return (perdidos)

quotation_values(df)


# In[9]:


perdidos = {}
for col in list(df.columns):
    if (df[col]).dtype == object:
        quotation = np.sum(df[col] == '?')
        perdidos[col] = quotation
perdidos = pd.DataFrame.from_dict(perdidos, orient = 'index')
print(perdidos)


# Los campos con valores "?" son tres, los cuales son **collision_type**, **property_damage**, **police_report_availabe**. Procedemos a reemplazar los valores "?" por "nan" para luego optar por imputar o no los valores.

# In[10]:


# reemplazamos los valores con "?" en el dataset
df.replace('?', np.nan, inplace = True)


# Veamos la proporcion de datos perdidos en cada variable

# In[11]:


prop_perdidos = df.isnull().sum() * 100 / len(df)
valores_perdidos_df = pd.DataFrame({'Variable': df.columns,
                                 '% de perdidos': prop_perdidos})

valores_perdidos_df


# Procedemos a imputar los valores perdidos en las variables que poseen nulos, usando imputacion basada en la categoria mas frecuente.

# In[12]:


#Seleccionamos los campos a imputar
fields_impute = df[["property_damage", 'police_report_available', 'collision_type']]

# guardamos los nombres de las columnas
col_names_impute = list(fields_impute.columns)

#extraemos el array con los valores de las variables escogidas
data = fields_impute.values
data.shape

#https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
#https://dzone.com/articles/imputing-missing-data-using-sklearn-simpleimputer#:~:text=You%20can%20use%20Sklearn.,and%20constant%20can%20be%20used


# In[13]:


print(fields_impute['police_report_available'].value_counts())
print(fields_impute['collision_type'].value_counts())
print(fields_impute['property_damage'].value_counts())


# In[14]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(data)
data_imputed = imputer.transform(data)
data_imputed.shape


# In[15]:


# creamos el dataframe con los datos imputados
data_imputed_df = pd.DataFrame(data_imputed, columns = col_names_impute)
data_imputed_df.head()


# In[16]:


# concatenamos los datos originales con los imputados
df.drop(['property_damage', 'police_report_available', 'collision_type'], axis = 1, inplace = True)
df = pd.concat([df, data_imputed_df], axis = 1)

#Verficamos que los datos se han imputado correctamente
df.isna().sum()


# In[17]:


print(df['police_report_available'].value_counts())
print(df['collision_type'].value_counts())
print(df['property_damage'].value_counts())


# In[18]:


df.head()


# In[19]:


df.describe()


# Hemos imputado los valores faltantes de manera satisfactoria.

# ## 3. Outliers en la data 

# Seleccionamos los campos de tipo numerico y categorico y los guardamos en una variable llamada **data_outliers**

# In[20]:


numeric_df = df.select_dtypes(include = ["number"])


# In[21]:


#Hacemos un drop de la columna "policy_number" e insured_zip ya que es un index la primera y la segunda es una locacion.
numeric_df.drop(['policy_number', 'insured_zip'], axis=1, inplace=True)


# Usar clases para construir modelos
# https://towardsdatascience.com/using-classes-for-machine-learning-2ed6c0713305

# Veamos la distribucion de los datos en cada variable numerica 

# In[22]:


plt.figure(figsize = (25, 20))
plotnumber = 1

for col in numeric_df.columns:
    if plotnumber <= 28:
        ax = plt.subplot(5, 5, plotnumber)
        sns.distplot(numeric_df[col])
        plt.xlabel(col, fontsize = 15)
        
    plotnumber += 1
    
plt.tight_layout()
plt.show()


# Veamos ahora los outliers en los campos numericos   

# In[23]:


plt.figure(figsize = (20, 15))
plot_box = 1

for col in numeric_df.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plot_box)
        sns.boxplot(numeric_df[col])
        plt.xlabel(col, fontsize = 15)
    
    plot_box += 1
plt.tight_layout()
plt.show()


# Los outliers parece que son datos que pueden ser reales en el contexto de los casos, podriamos ver la distribucion de los casos por la variable objetivo para tener mejor idea de lo que pasa. Podemos hacer un escalamiento de las variables numricas luego, antes de realizar la modelacion.

# ## 4. Visualizacion datos
# 

# ### 4.1 Analisis Exploratorio de datos

# In[24]:


Baseline=pd.DataFrame({'Count':df.groupby(['fraud_reported']).size()})
Baseline=Baseline.reset_index()
Baseline['Prop']=Baseline['Count']/Baseline['Count'].sum()


# In[25]:


Baseline


# In[26]:


(ggplot(Baseline, aes(x='fraud_reported',y='Prop',fill='fraud_reported'))+geom_col()+ggtitle('Baseline'))


# In[27]:


# creamos una lista con las ciudades ordenadas de mayor a menor en la cantidad o conteo
city = df['incident_city'].value_counts().index.to_list()[::-1]

ggplot(df) + geom_bar(aes(x='incident_city', fill='fraud_reported'))+\
scale_x_discrete(limits=city)+\
    ggtitle('Ciudad del incidente vs fraude')


# In[28]:


# creamos la lista ordenada de las marcas
make = df.auto_make.value_counts().index.to_list()[::-1]

ggplot(df) + geom_bar(aes(x='auto_make', fill='fraud_reported'))+\
coord_flip() + scale_x_discrete(limits=make)+\
     ggtitle('Marca del auto vs fraude')


# In[29]:


def highlight(val):
  return ['background-color: red' if a>0.7 else '' for a in val] 
  
make=pd.crosstab(df.auto_make, df.fraud_reported ,normalize="index").transpose()
make.style.apply(highlight)


# Se observa una mayor cantidad de autos de la marca Mercedes Benz envueltos en reclamaciones fraudulentas. Los Nissan parecen tener la menor propocion de casos fraudulentos dentro de su grupo.

# In[30]:


ggplot(df) + geom_bar(aes(x='auto_year', fill='fraud_reported'))


# In[31]:


ggplot(df) + geom_bar(aes(x='policy_state', fill='fraud_reported'))+ggtitle('Estado de la poliza (lugar) vs Fraude')


# In[32]:


ggplot(df) + geom_bar(aes(x='insured_sex', fill='fraud_reported'))


# In[33]:


gender=pd.crosstab(df.insured_sex, df.fraud_reported ,normalize="index").transpose()
#gender.style.apply(highlight)
gender


# In[34]:


ggplot(df) + geom_bar(aes(x='policy_csl', fill='fraud_reported'))


# In[35]:


# creamos una lista con las ocupaciones ordenadas de mayor a menor en la cantidad o conteo
occupation = df['insured_occupation'].value_counts().index.to_list()[::-1]

# se usa coord_flip para cambiar a horizontal el grafico
ggplot(df) + geom_bar(aes(x='insured_occupation', fill='fraud_reported'))+\
coord_flip() + scale_x_discrete(limits=occupation)+\
     ggtitle('Ocupacion del asegurado vs fraude')


# In[36]:


def highlight(val):
  return ['background-color: red' if a>0.7 else '' for a in val] 
  
occupation=pd.crosstab(df.insured_occupation, df.fraud_reported ,normalize="index").transpose()
occupation.style.apply(highlight)


# In[37]:


education = df['insured_education_level'].value_counts().index.to_list()[::-1]
ggplot(df) + geom_bar(aes(x='insured_education_level', fill='fraud_reported')) +\
     scale_x_discrete(limits=education)+\
     ggtitle('Educacion vs fraude')


# Las personas de menor nivel educativo son mas propensas a realizar fraude en las reclamaciones. por el contrario, a mayor nivel educativo, menor es la probabilidad de realizar fraude.

# In[38]:


ggplot(df) + geom_bar(aes(x='collision_type', fill='fraud_reported')) +\
          ggtitle('Tipo Colision vs fraude')


# ### Visualizacion de los campos numericos

# ### 4.2 Correlacion de los datos

# Observemos ahora como esta la correlacion entre las variables numericas del dataset

# In[39]:


plt.figure(figsize = (18, 12))

corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))

sns.heatmap(data = corr, cmap="viridis", mask = mask, annot = True, fmt = '.2g', linewidth = 1)
plt.show()


# In[40]:


#plt.figure(figsize=[10,10])

corr_df = df[['months_as_customer', 'age','total_claim_amount', 'injury_claim','property_claim', 'vehicle_claim','fraud_reported']]
#corr_df = df[['total_claim_amount', 'injury_claim','property_claim', 'vehicle_claim','fraud_reported']]
sns.pairplot(corr_df,hue='fraud_reported',corner=True)
#plt.show()


# Se puede descartar las variables que tengan alta correlacion entre ellas y asi evitar la colinealidad en la data.
# 
# Por ejemplo **age** del cliente se correlaciona fuertemente con la variable **months_as_customer**, podemos descartar la antiguedad en este caso. sucede lo mismo con las variables **total_claim_amount**, que se relaciona fuertemente con las variables **injury_claim**, **property_claim** y **vehicle_claim** asi que esta primera podria dejarse en la data, ya que recoge la informacion de las demas variables y asi evitar la colinealidad al momento de la modelacion.
# 
# En este sentido, podemos crear una funcion para descartar los campos que posean alta correlacion o por encima de umbral especifico, en este caso 0.8.

# In[41]:


# create a function to detect variables with correlation superior to 0.8
def correl_drop(df, umbral):
    correlation = df.corr().abs()
    upper_matrix = correlation.where(np.triu(np.ones(correlation.shape), k = 1).astype(np.bool))
    drop_cols = [column for column in upper_matrix.columns if any(upper_matrix[column] > umbral)]
    return drop_cols

correl_drop(numeric_df, 0.5)


# En este caso, descartaremos las variables **months_as_customer**, **injury_claim**, **property_claim** y **vehicle_claim**, pero esto lo haremos mas adelante.

# Veamos algunos graficos para evidenciar el comportamiento de las variables vs la variable objetivo.

# In[42]:


ggplot(aes(x=df.vehicle_claim,color=df.fraud_reported,fill = df.fraud_reported))+geom_histogram()+ggtitle('monto reclamacion vs fraude')


# In[43]:


ggplot(aes(x=df.vehicle_claim,color=df.fraud_reported))+geom_density()+ggtitle('monto reclamacion vs fraude')


# In[44]:


ggplot(aes(y=df.vehicle_claim,x=df.fraud_reported, fill = df.fraud_reported))+geom_boxplot()+ggtitle('monto reclamacion vs fraude')


# In[45]:


shift = 0.1
(ggplot(aes(y=df.vehicle_claim,x=df.fraud_reported, fill = df.fraud_reported))
 + geom_violin()+ geom_boxplot(width = shift)
)


# In[46]:


ggplot(df) + geom_bar(aes(x='auto_year', fill='fraud_reported'))+ggtitle('Año del auto vs fraude')


# In[47]:


np.min(df.policy_bind_date)
np.max(df.policy_bind_date)


# In[48]:


date_policy = pd.crosstab(df.policy_bind_date, df.fraud_reported)
date_policy.head()


# El campo **policy bind date** no muestra una relacion importante con la variable objetivo 

# In[49]:


hobbies = df['insured_hobbies'].value_counts().index.tolist()[::-1]
ggplot(df) + geom_bar(aes(x='insured_hobbies', fill='fraud_reported'))+\
    coord_flip()+ scale_x_discrete(limits=hobbies)+\
    ggtitle('Hobbies del asegurado vs fraude')


# Las personas que juegan *ajedrez* o realizan *crossfit* en sus **hobbies**, estan mas propensas a cometer fraude, por lo que es interesante este campo en la prediccion de los casos de fraude.

# In[50]:


# ordeanmos los datos por la frecuencia de las categorias de ocupacion
ocupation = df['insured_occupation'].value_counts().index.tolist()[::-1]
# generamos el grafico
ggplot(df) + geom_bar(aes(x='insured_occupation', fill='fraud_reported'))+\
    coord_flip()+scale_x_discrete(limits=ocupation) +\
        ggtitle('Ocupacion del asegurado vs fraude')


# In[51]:


relation = df['insured_relationship'].value_counts().index.tolist()[::-1]
# generamos el grafico
ggplot(df) + geom_bar(aes(x='insured_relationship', fill='fraud_reported'))+\
    scale_x_discrete(limits=relation) +\
        ggtitle('Relacion del asegurado vs fraude')


# In[52]:


severity = df['incident_severity'].value_counts().index.tolist()[::-1]
# generamos el grafico
ggplot(df) + geom_bar(aes(x='incident_severity', fill='fraud_reported'))+\
    scale_x_discrete(limits=severity) +\
        ggtitle('Severidad del incidente vs fraude')


# In[53]:


ggplot(df) + geom_bar(aes(x='incident_type', fill='fraud_reported'))+\
           ggtitle('Tipo de incidente vs fraude')


# In[54]:


ggplot(df) + geom_bar(aes(x='authorities_contacted', fill='fraud_reported'))+\
        ggtitle('Autoridades contactadas vs fraude')


# In[55]:


ggplot(df) + geom_bar(aes(x='authorities_contacted', fill='fraud_reported'))+\
           ggtitle('Relacion del asegurado vs fraude')


# In[56]:


auto_model = df['auto_model'].value_counts().index.tolist()[::-1]
ggplot(df) + geom_bar(aes(x='auto_model', fill='fraud_reported'))+\
    coord_flip()+ scale_x_discrete(limits=auto_model)+\
    ggtitle('Modelo del auto vs fraude')


# In[57]:


auto_model=pd.crosstab( df.auto_model, df.fraud_reported ,normalize="index").transpose()
auto_model.style.apply(highlight)


# In[58]:


ggplot(aes(x=df.age,color=df.fraud_reported))+geom_density()+ggtitle('edad vs fraude')


# In[59]:


shift = 0.1
(ggplot(aes(y=df.age,x=df.fraud_reported, fill = df.fraud_reported))
 + geom_violin()+ geom_boxplot(width = shift))


# In[60]:


shift = 0.1
(ggplot(aes(y=df.months_as_customer,x=df.fraud_reported, fill = df.fraud_reported))
 + geom_violin()+ geom_boxplot(width = shift))


# In[61]:


ggplot(aes(x=df.policy_deductable,color=df.fraud_reported))+geom_density()+ggtitle('edad vs fraude')


# In[62]:


shift = 0.1
(ggplot(aes(y=df.policy_annual_premium,x=df.fraud_reported, fill = df.fraud_reported))
 + geom_violin()+ geom_boxplot(width = shift))


# In[63]:


ggplot(aes(x=df.policy_annual_premium,color=df.fraud_reported))+geom_density()+ggtitle('Prima poliza anual vs fraude')


# In[64]:


ggplot(aes(x=df[['capital-gains']],color=df.fraud_reported))+geom_density()+ggtitle('Ganancia de capital vs fraude')


# In[65]:


shift = 0.1
(ggplot(aes(y=df[['capital-gains']],x=df.fraud_reported, fill = df.fraud_reported))
 + geom_violin()+ geom_boxplot(width = shift))


# In[66]:


ggplot(aes(x=df[['capital-loss']],color=df.fraud_reported))+geom_density()+ggtitle('Ganancia de capital vs fraude')


# In[67]:


ggplot(aes(x=df.injury_claim,color=df.fraud_reported))+geom_density()+ggtitle('Ganancia de capital vs fraude')


# In[68]:


shift = 0.1
(ggplot(aes(y=df.injury_claim,x=df.fraud_reported, fill = df.fraud_reported))
 + geom_violin()+ geom_boxplot(width = shift))


# In[69]:


shift = 0.1
(ggplot(aes(y=df.property_claim,x=df.fraud_reported, fill = df.fraud_reported))
 + geom_violin()+ geom_boxplot(width = shift))+ggtitle('Valor reclamacion propiedad vs fraude')


# In[70]:


shift = 0.1
(ggplot(aes(y=df.injury_claim,x=df.fraud_reported, fill = df.fraud_reported))
 + geom_violin()+ geom_boxplot(width = shift))+ggtitle('Valor reclamacion lesiones vs fraude')


# In[71]:


shift = 0.1
(ggplot(aes(y=df.vehicle_claim,x=df.fraud_reported, fill = df.fraud_reported))
 + geom_violin()+ geom_boxplot(width = shift))+ggtitle('Valor reclamacion lesiones vs fraude')


# ## 4. Preprocesamiento de datos

# ### 4.1 Creacion de nuevas features

# Empezaremos con crear un campo que indica la cantidad de anios que el vehiculo posee, usando la columna del anio en que se manufacturo el vehiculo **(auto_year)** esto nos servira para identificar si la antiguedad del modelo del vehiculo posee alguna relacion con el fraude o no.

# In[72]:


anio_actual = pd.datetime.now().year # obtenemos el anio actual
df['age_vehicle'] = anio_actual - df['auto_year'] # calculamos la edad del vehiculo

df['age_vehicle'].head(10)


# In[73]:


ggplot(aes(x=df.age_vehicle,color=df.fraud_reported, fill = df.fraud_reported))+geom_bar()+ggtitle('Edad del vehiculo vs fraude')


# Veamos las proporciones para cada anio del vehiculo

# In[74]:


def highlight(val):
  return ['background-color: red' if a>0.7 else '' for a in val] 

# Crosstab por anio del vehiculo
ct=pd.crosstab(df.age_vehicle,df.fraud_reported,normalize="index").transpose()
ct.style.apply(highlight)


# Ahora vamos a hacer un split en el campo de policy_csl, ya que en la teoria este es el monto de cobertura d ela poliza por danos a persona y danos a vehiculo respectivamente.

# In[75]:


df['csl_person'] = df.policy_csl.str.split('/', expand=True)[0]
df['csl_accident'] = df.policy_csl.str.split('/', expand=True)[1]
df[['policy_csl', 'csl_person', 'csl_accident']].head()


# Ahora convertimos el campo de hora de ocurrencia del accidente y lo categorizamos dependiendo la hora del dia. podemos dividir el dia de 24 hrs en intervalos de 3 horas, dando como resutado 8 grupos de horas.

# In[76]:


# Creamos los intervalos en grupos de 3 hrs
intervalos = [-1, 3, 6, 9, 12, 16, 20, 24]  
# asignamos los nombres a los intervalos de tiempo
cat = ["medianoche", "manana_temprano", "manana", 'mediodia', 'tarde', 'noche_temprano', 'noche']
df['incidente_periodo_dia'] = pd.cut(df.incident_hour_of_the_day, intervalos, labels=cat).astype(object)
df[['incident_hour_of_the_day', 'incidente_periodo_dia']].head(20)


# Veamos graficamente como se distribuyen los casos por periodo del dia de ocurrencia del accidente. Parece haber maor cantidad de casos en la tarde fraudulentos mientras que en la manana son menos frecuentes.

# In[77]:


incidente_hora = df.incidente_periodo_dia.value_counts().index.tolist()[::-1]

# generamos el grafico
ggplot(df) + geom_bar(aes(x='incidente_periodo_dia', fill='fraud_reported'))+\
    coord_flip()+scale_x_discrete(limits=incidente_hora) +\
        ggtitle('Periodo dia vs fraude')


# In[78]:


# Crosstab por incidente_periodo_dia del vehiculo
ct1=pd.crosstab(df.incidente_periodo_dia,df.fraud_reported,normalize="index").transpose()
ct1.style.apply(highlight)


# El campo **umbrella_limit** posee un valor negativo que no es correcto segun el contexto de la variable, se reemplaza por un valor 0.

# In[79]:


df['umbrella_limit']=df['umbrella_limit'].replace(-1000000,0)


# Hacemos un drop de las columnas que no se utilizaran mas en la data. Se incluyen las columnas con alta correlacion como age, injury_claim, vehicle_claim, property_claim ademas de las columnas de fechas y de localizacion de incidente ya qe tienen muchas categorias que no se pueden agrupar con facilidad.

# In[80]:


# Drop de columnas:

df = df.drop(columns = [
    'policy_csl',
    'insured_zip',
    'policy_bind_date', 
    'incident_date', 
    'incident_location', 
    'auto_year', 
    'policy_number',
    'injury_claim',
    'vehicle_claim',
    'property_claim',
    'months_as_customer',
    'incident_hour_of_the_day'])


# In[81]:


fig = px.choropleth(df,
                    locations='incident_state', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='total_claim_amount',
                    color_continuous_scale="Viridis_r", 
                    
                    )
fig.show()


# In[82]:


df.head()


# ## 5. Creacion datasets de train y test

# Realizamos un split de los datos, las variables independientes y las variables dependientes, ademas convertimos la variable dependiente **fraud_reported** en binaria

# In[83]:


# variables independientes:
X = df.drop(['fraud_reported'], axis=1)
# variable dependiente y la binarizamos usando el metod where de numpy
y = np.where(df['fraud_reported'] == 'Y', 1, 0)


# Antes de realizar el split de los datos, aplicamos one-hot econding para las variables categoricas y asi binarizar el dataset

# ### 5.2 One-Hot Encoding en las variables categoricas

# Creamos el dataset de variables categoricas y lo guardamos en una variable llamada **categorical_df**, para luego aplicarle la funcion **get_dummies**

# In[84]:


categorical_df = X.select_dtypes(include = ["object"])

# creamos las variables dummy
categorical_df_dummy = pd.get_dummies(categorical_df, drop_first=True)
#categorical_df_dummy = pd.get_dummies(categorical_df)
categorical_df_dummy.shape


# In[85]:


categorical_df_dummy.head()


# Agrupamos las variables para crear nuevamente el dataset completo

# In[86]:


X1 = categorical_df_dummy.join(X.select_dtypes(include = ['number']))
names_cols = X1.columns.tolist() # Guardamos los nombres de las columnas para luego aplicar algoritmo boruta
X1.head()
print(X1.shape)


# ### 5.3 Feaure selection

# #### Boruta Algorithm

# Vamos a utilizar el algorimto Boruta para la seleccion de las variables, en este caso lo aplicamos utilizando la libreria **Borutapy**

# In[87]:


# TEST  Boruta algorithm
# fit the random forest model

#forest = RandomForestClassifier(n_estimators=100, random_state=0)

# convert df to numpy array
X = X1.copy().values

# define random forest classifier
forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
forest.fit(X, y)


# In[88]:


# define Boruta feature selection method
feat_selector = BorutaPy(forest, n_estimators='auto', verbose=1, random_state=1)

# find all relevant features
feat_selector.fit(X, y)

# check selected features
feat_selector.support_

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)


# In[89]:


# zip my names, ranks, and decisions in a single iterable
feature_ranks = list(zip(names_cols, 
                        feat_selector.ranking_, 
                        feat_selector.support_))

# iterate through and print out the results
for feat in feature_ranks:
    print('Feature: {:<25} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))


# ## 5. Creacion conjuntos de Train y Test / Balanceo de datos 

# La variable objetivo es la variable **fraud_reported**, la cual tiene una distribucion de valores de 1 a 0, donde 1 es fraude y 0 No hay fraude. Usaremos el algoritmo SMOTE para balancear los datos y se probaran algunos otros metodos como oversampling.

# In[90]:


smote = SMOTE(random_state=42)


# In[91]:


X_train, X_test, y_train, y_test = train_test_split(
    X,  # matriz con las variables predictivas
    y, # array con los valores de la variable objetivo
    test_size=0.2,  # proporción a dejar en el test set
    random_state=123, # para controlar la semilla aleatoria
    stratify=y) # indica la variable de estratificación estratificación de la muestra


# In[92]:


# summarize class distribution original Y and Y_train
counter = Counter(y)
counter_ytrain = Counter(y_train)
print(counter)
print(counter_ytrain)


# In[93]:


# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = np.where(y == label)[0]
	plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
plt.legend()
plt.show()


# In[94]:


### este es el original

from imblearn.over_sampling import RandomOverSampler
from collections import Counter
'''
rus = RandomOverSampler(random_state=4,sampling_strategy=1)
X_train_r, y_train_r = rus.fit_resample(X_train,y_train)
###la data es devuelta en forma secuencial: una clase tras otra, por lo que debemos aleatorizarla
data_r=pd.DataFrame(np.column_stack([X_train_r,y_train_r])).sample(frac=1).values
print("Valores balanceados : ", Counter(data_r[:,-1]));

nrow,ncol=data_r.shape
X_train_r=data_r[:,0:(ncol-1)]
y_train_r=data_r[:,-1]
y_train=y_train_r
'''


# In[95]:


#print(X_train_r.shape)
#print(y_train.shape)


# In[96]:


pd.Series(y_train).value_counts().plot(kind='bar', title='Clases variable objetivo antes de aplicar SMOTE', xlabel='fraud reported')


# In[97]:


#### Otra implementacion usando SMOTE

#sm = SMOTE(random_state=4, sampling_strategy=0.5)
sm = SMOTE(random_state=4, sampling_strategy=1)

X_train_r, y_train_r = sm.fit_resample(X_train,y_train)
###la data es devuelta en forma secuencial: una clase tras otra, por lo que debemos aleatorizarla
#data_r=pd.DataFrame(np.column_stack([X_train_r,y_train_r])).sample(frac=1).values
#print("Valores balanceados : ", Counter(data_r[:,-1]));

counter = Counter(y_train_r)
print(counter)

#nrow,ncol=data_r.shape
#X_train_r=data_r[:,0:(ncol-1)]
#y_train_r=data_r[:,-1]
#y_train=y_train_r

y_train = y_train_r


# Realizamos un grafico para comprobar el resultado luego de aplicar SMOTE en la variable objetivo **fraud reported**

# In[98]:


pd.Series(y_train_r).value_counts().plot(kind='bar', title='Clases variable objetivo luego de aplicar SMOTE', xlabel='fraud reported')


# In[99]:


# vemos las formas de los conjuntos de datos
print(X_train.shape)
print(y_train.shape)

print(X_train_r.shape)
print(X_test.shape)

print(y_train_r.shape)
print(y_test.shape)


# # Estandarizacion de Datos:

# ### 5.3 Estandarizacion de las variables numericas

# Como se vio el los graficos de boxplot, existen campos numericos con distintas magnitudes, por lo que procedemos a estandarizar los datos numericos para que sean comparables en los modelos y no provoque un sesgo en las predicciones. Para este caso, se usara el StandardScaler

# In[100]:


# creamos el scaler
#scaler = StandardScaler()
scaler = MinMaxScaler()

# renombramos los datasets creados anteriomente
DataX_train=X_train_r
DataX_test=X_test

#escalamos los conjuntos de datos
DataX_train=scaler.fit_transform(DataX_train)
DataX_test=scaler.transform(DataX_test)


# In[101]:


DataX_train


# In[102]:


DataX_test


# In[103]:


#Comprobamos las dimensiones de los datasets
print(DataX_train.shape)
print(DataX_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### BORUTA FEATURE SELECTION

# In[104]:


# TEST  Boruta algorithm
# fit the random forest model

#forest = RandomForestClassifier(n_estimators=100, random_state=0)

# convert df to numpy array
X = DataX_train
y = y_train
# define random forest classifier
forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=44)
forest.fit(X, y)


# define Boruta feature selection method
feat_selector = BorutaPy(forest, n_estimators='auto', verbose=1, random_state=1)

# find all relevant features
feat_selector.fit(X, y)

# check selected features
feat_selector.support_

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)


# In[105]:


# Creamos el dataset con la data

DT = DecisionTreeClassifier(min_samples_split=45, max_depth = 4)
DT_model = DT.fit(X_filtered, y_train)

X_test_filtered = feat_selector.transform(DataX_test)

prediction_DT = DT_model.predict(X_test_filtered)


# In[106]:


print("accuracy = ", accuracy_score(y_test, prediction_DT))
print(recall_score(y_test, prediction_DT))
print(f1_score(y_test, prediction_DT))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_DT)

#fig, ax = plt.subplots(figsize=(10,10))  
sns.heatmap(cm, annot = True, fmt='g')


# In[107]:


X_filtered.shape


# ### OTROS METODOS DE SELECCION FEATURES

# ### RFE

# In[108]:


from sklearn.feature_selection import RFE
model = DecisionTreeClassifier()
rfe = RFE(estimator = model, n_features_to_select = 5)
fit = rfe.fit(DataX_train, y_train)


# In[109]:


print("num features: %d" % fit.n_features_)
print('Selected features: %s' % fit.support_)
print('feature ranking: %s' % fit.ranking_)


# In[110]:


model.fit(DataX_train, y_train)


# En este sentido, vemos que los resultados de las variables obtenidas tanto por RFE y Bortua son muy similares, siendo el algoritmo Boruta el mas parsimonioso los resultados mas 

# Vamos a usar las columnas transformadas para crear el dataset de train y test final para X, usando la seleccion realizada por el algoritmo de Boruta

# In[111]:


DataX_train = feat_selector.transform(DataX_train)
DataX_test = feat_selector.transform(DataX_test)

#Vemos la forma de los dataset finales
print(DataX_train.shape)
print(DataX_test.shape)


# ## 6. Entrenamiento y prueba de modelos

# Para el entrenamiento de los modelos, se propone crear una funcion que permita medir diversos scores que nos hagan mas facil la seleccion del modelo, teniendo en cuenta que se esta trabajando con datos desbalanceados, por lo que en este caso las metricas a usar son el accuracy, recall, precision, el F1-score, el coeficiente de  kappa, coeficiente de jaccard y el ROC score o area bajo la curva.

# #### Performance del modelo

# In[112]:


def metrics(real,pred):
  kappa=cohen_kappa_score(real,pred)
  acc=accuracy_score(real,pred)
  f1=f1_score(real,pred)
  prec=precision_score(real,pred)
  recall=recall_score(real,pred)
  jaccard=jaccard_score(real,pred)
  logloss=log_loss(real,pred)
  roc_score = roc_auc_score(real,pred)

  print (f" Accuracy:{acc:.4f} \n Precision: {prec:.4f} \n Recall: {recall:.4f} \n Kappa: {kappa:.4f} \n F1-Score: {f1:.4f} \n Jaccard: {jaccard:.4f} \n logloss: {logloss:.4f} \n Roc_score: {roc_score:.4f}")


# Ya con esta formula, procedemos a ajustar los modelos de clasificacion para el proyecto.

# ## 6.1  Entrenamiento de modelos

# ## Logistic Regression

# In[113]:


pipeline = Pipeline([
  ('clf', LogisticRegression(penalty='l2',class_weight='balanced', dual=False,
                                    fit_intercept=True, intercept_scaling=1,
                                    l1_ratio=None, max_iter=10000,
                                    multi_class='auto', n_jobs=None,
                                    random_state=4,
                                    solver='lbfgs', tol=0.0001, verbose=0,
                                    warm_start=False))])

params = {'clf__C':[0.001,0.01,0.1,1,10]}
scoring={'kappa':make_scorer(cohen_kappa_score),'accuracy':'accuracy'}
grid= GridSearchCV(pipeline, params,scoring=scoring,refit='kappa')
grid.fit(DataX_train, y_train)
pred_lass=grid.predict(DataX_test)
pred_lass_train=grid.predict(DataX_train)
print(grid.best_params_)

metrics(y_test,pred_lass)


# In[114]:


from sklearn.metrics import classification_report
print(classification_report(y_test, pred_lass))


# ### KNN

# In[115]:


clf = KNeighborsClassifier()
param_grid = {'n_neighbors': range(1, 100)}
grid1 = GridSearchCV(clf, param_grid=param_grid, cv = 5, scoring = scoring, n_jobs=-1, refit='kappa')
grid1.fit(DataX_train, y_train)
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
print(grid1.best_params_)
pred_knn_train=grid1.predict(DataX_train)
pred_knn=grid1.predict(DataX_test)
prob_knn_train=grid1.predict_proba(DataX_train)
prob_knn_test=grid1.predict_proba(DataX_test)
metrics(y_test,pred_knn)


# ## Decision tree

# In[116]:


## definir pasos para estandarizar los datos y entrenar el Decision Tree
from sklearn.tree import DecisionTreeClassifier

steps = [('DT', DecisionTreeClassifier(random_state=4))]
pipeline = Pipeline(steps)
parametros = {'DT__max_depth': range(2,20), "DT__min_samples_split": [2, 10, 20, 30, 40, 45, 50, 100]}
grid1 = GridSearchCV(pipeline, param_grid=parametros, cv = 5, scoring = scoring, n_jobs=-1, refit='kappa')
grid1.fit(DataX_train, y_train)
##correr el pipeline
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
print(grid1.best_params_)
pred_dt_train=grid1.predict(DataX_train)
pred_dt=grid1.predict(DataX_test)
prob_dt_train=grid1.predict_proba(DataX_train)
prob_dt_test=grid1.predict_proba(DataX_test)
metrics(y_test,pred_dt)


# ##  Random forest
# https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/

# In[117]:


steps = [('RF', RandomForestClassifier(random_state=4, n_jobs=-1))]
pipeline = Pipeline(steps)
#parametros = {'RF__n_estimators':[5,10,15,20,50,100,500,1000], 'RF__min_samples_split':[2,3, 4,5,10,20,50,100], 'RF__max_depth':range(2,20)}
parametros = {'RF__n_estimators':[5,10,15,20,50,100, 1000], 'RF__min_samples_split':[2,3, 4,5,10,20,50,100, 150], 'RF__max_depth':[2,3,5,10,20], 'RF__min_samples_leaf': [5,10,20,50,100,200]}
grid1 = GridSearchCV(pipeline, param_grid=parametros, cv=5, scoring=scoring,n_jobs=-1,refit='kappa')
grid1.fit(DataX_train, y_train)
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
print(grid1.best_params_)
pred_rf_train=grid1.predict(DataX_train)
pred_rf=grid1.predict(DataX_test)
prob_rf_train=grid1.predict_proba(DataX_train)
prob_rf_test=grid1.predict_proba(DataX_test)
RF = metrics(y_test,pred_rf) 


# ## Linear Discriminat Analyses (LDA)

# In[118]:


## definir pasos para estandarizar los datos y entrenar el LDA
steps = [('LDA', LinearDiscriminantAnalysis())]
pipelineLDA = Pipeline(steps)
##correr el pipeline
pipelineLDA.get_params().keys()
pipelineLDA.fit(DataX_train, y_train)
pred_lda_train=pipelineLDA.predict(DataX_train)
pred_lda=pipelineLDA.predict(DataX_test)
prob_lda_train=pipelineLDA.predict_proba(DataX_train)
prob_lda_test=pipelineLDA.predict_proba(DataX_test)
LDA_met = metrics(y_test,pred_lda)
#metrics(y_test,pred_lda)


# ## Quadratic Discriminat Analyses  (QDA)

# In[119]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
steps = [('QDA', QuadraticDiscriminantAnalysis())]
pipelineQDA = Pipeline(steps)
##correr el pipeline
pipelineQDA.get_params().keys()
pipelineQDA.fit(DataX_train, y_train)
pred_qda=pipelineQDA.predict(DataX_test)
pred_qda_train=pipelineQDA.predict(DataX_train)
metrics(y_test,pred_qda)


# ## SVM

# Kernel Lineal

# In[120]:


scoring={'kappa':make_scorer(cohen_kappa_score),'accuracy':'accuracy'}
steps = [("SVM_linear", SVC(kernel="linear",probability=True, random_state=4))]
pipeline = Pipeline(steps)
parametros = {'SVM_linear__C':[0.01,0.1,1,10]}
grid1 = GridSearchCV(pipeline, param_grid=parametros, cv=5, scoring=scoring,n_jobs=-1,refit='kappa')
grid1.fit(DataX_train, y_train)
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
print(grid1.best_params_)
y_svm_lin=grid1.predict(DataX_test)
y_svm_lin_train=grid1.predict(DataX_train)
prob_svm_lin=grid1.predict_proba(DataX_test)
metrics(y_test,y_svm_lin)


# Kernel RBF

# In[121]:


steps = [("SVM_rbf", SVC(kernel="rbf",probability=True, random_state=4))]
pipeline = Pipeline(steps)
parametros = {'SVM_rbf__C':[0.01,0.1,1,10], 'SVM_rbf__gamma':[0.05,0.01, 1, 5]}
grid1 = GridSearchCV(pipeline, param_grid=parametros, cv=5, scoring=scoring,n_jobs=-1,refit='kappa')
grid1.fit(DataX_train, y_train)
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
print(grid1.best_params_)
y_svm_rbf=grid1.predict(DataX_test)
y_svm_rbf_train=grid1.predict(DataX_train)
prob_svm_rbf_prob=grid1.predict_proba(DataX_train)
metrics(y_test,y_svm_rbf)
prob_svm_rbf=grid1.predict_proba(DataX_test)


# Kernel Sigmoid

# In[122]:


steps = [("SVM_sigmoid", SVC(kernel="sigmoid",probability=True, random_state=4))]
pipeline = Pipeline(steps)
parametros = {'SVM_sigmoid__C':[0.01,0.1,1,10], 'SVM_sigmoid__gamma':[0.05,0.01, 1, 5]}
grid1 = GridSearchCV(pipeline, param_grid=parametros, cv=5, scoring=scoring,n_jobs=-1,refit='kappa')
grid1.fit(DataX_train, y_train)
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
y_svm_sig=grid1.predict(DataX_test)
y_svm_sig_train=grid1.predict(DataX_train)
print(grid1.best_params_)
prob_svm_sig=grid1.predict_proba(DataX_test)
metrics(y_test,y_svm_sig)


# Ahora veamos si una descomposición en Componentes Principales de los valores,ayuda en la predicción: 

# In[123]:


steps = [('PCA', PCA()), ("SVM_linear", SVC(kernel="linear",probability=True, random_state=4))]
pipeline = Pipeline(steps)
parametros = {'SVM_linear__C':[0.01,0.1,1,10],'PCA__n_components':[None,2, 3, 4,5,7]}
grid1 = GridSearchCV(pipeline, param_grid=parametros, cv=5, scoring=scoring,n_jobs=-1,refit='kappa')
grid1.fit(DataX_train, y_train)
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
print(grid1.best_params_)
y_svm_lin_pca=grid1.predict(DataX_test)
y_svm_lin_pca_train=grid1.predict(DataX_train)
prob_lin_sig_pca=grid1.predict_proba(DataX_train)
prob_lin_sig_pca_test=grid1.predict_proba(DataX_test)
metrics(y_test,y_svm_lin_pca)


# In[124]:


steps = [('PCA', PCA()),("SVM_rbf", SVC(kernel="rbf",probability=True, random_state=4))]
pipeline = Pipeline(steps)
parametros = {'SVM_rbf__C':[0.01,0.1,1,10], 'SVM_rbf__gamma':[0.05,0.01, 1, 5], 'PCA__n_components':[None,2, 3, 4,5,7]}
grid1 = GridSearchCV(pipeline, param_grid=parametros, cv=5, scoring=scoring,n_jobs=-1,refit='kappa')
grid1.fit(DataX_train, y_train)
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
print(grid1.best_params_)
y_svm_rbf_pca=grid1.predict(DataX_test)
y_svm_rbf_pca_train=grid1.predict(DataX_train)
prob_svm_rbf_pca=grid1.predict_proba(DataX_train)
prob_svm_rbf_pca_test=grid1.predict_proba(DataX_test)
metrics(y_test,y_svm_rbf_pca)


# **Kernel** - SVM Sigmoide con Reducción de Dimensionalidad

# In[125]:


from sklearn.decomposition import PCA
metrica='accuracy'
steps = [('PCA', PCA()),("SVM_sigmoid", SVC(kernel="sigmoid",probability=True, random_state=4))]
pipeline = Pipeline(steps)
parametros = {'SVM_sigmoid__C':[0.01,0.1,1,10], 'SVM_sigmoid__gamma':[0.05,0.01, 1, 5], 'PCA__n_components':[None, 2, 3, 4,5,7]}
grid1 = GridSearchCV(pipeline, param_grid=parametros, cv=5, scoring=scoring,n_jobs=-1,refit='kappa')
grid1.fit(DataX_train, y_train)
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
print(grid1.best_params_)
y_svm_sig_pca=grid1.predict(DataX_test)
y_svm_sig_pca_train=grid1.predict(DataX_train)
prob_svm_sig_pca=grid1.predict_proba(DataX_test)
metrics(y_test,y_svm_sig_pca)


# SVM usando LDA como reducción de dimensionalidad
# Kernel lineal

# In[126]:


steps = [('LDA', LinearDiscriminantAnalysis()), ("SVM_linear", SVC(kernel="linear",probability=True, random_state=4))]
pipeline = Pipeline(steps)
parametros = {'SVM_linear__C':[0.01,0.1,1,10]}
grid1 = GridSearchCV(pipeline, param_grid=parametros, cv=5, scoring=scoring,n_jobs=-1,refit='kappa')
grid1.fit(DataX_train, y_train)
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
print(grid1.best_params_)
y_svm_lin_pca=grid1.predict(DataX_test)
y_svm_lin_pca_train=grid1.predict(DataX_train)
prob_lin_sig_pca=grid1.predict_proba(DataX_train)
prob_lin_sig_pca_test=grid1.predict_proba(DataX_test)
metrics(y_test,y_svm_lin_pca)


# SVM usando LDA como reducción de dimensionalidad
# Kernel RBF

# In[127]:


metrica='accuracy'
steps = [('LDA', LinearDiscriminantAnalysis()),("SVM_rbf", SVC(kernel="rbf",probability=True, random_state=4))]
pipeline = Pipeline(steps)
parametros = {'SVM_rbf__C':[0.01,0.1,1,10], 'SVM_rbf__gamma':[0.05,0.01, 1, 5],'LDA__n_components':[None, 1]}
grid1 = GridSearchCV(pipeline, param_grid=parametros, cv=5, scoring=scoring,n_jobs=-1,refit='kappa')
grid1.fit(DataX_train, y_train)
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
print(grid1.best_params_)
y_svm_rbf_lda=grid1.predict(DataX_test)
y_svm_rbf_lda_train=grid1.predict(DataX_train)
prob_svm_rbf_lda=grid1.predict_proba(DataX_train)
prob_svm_rbf_lda_test=grid1.predict_proba(DataX_test)
metrics(y_test,y_svm_rbf_lda)


# SVM usando LDA como reducción de dimensionalidad
# Kernel sigmoide

# In[128]:


metrica='accuracy'
steps = [('LDA', LinearDiscriminantAnalysis()),("SVM_sigmoid", SVC(kernel="sigmoid",probability=True,  random_state=4))]
pipeline = Pipeline(steps)
parametros = {'SVM_sigmoid__C':[0.01,0.1,1,10], 'SVM_sigmoid__gamma':[0.05,0.01, 1, 5],'LDA__n_components':[None, 1]}
grid1 = GridSearchCV(pipeline, param_grid=parametros, cv=5, scoring=scoring,n_jobs=-1,refit='kappa')
grid1.fit(DataX_train, y_train)
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
print(grid1.best_params_)
y_svm_rbf_lda=grid1.predict(DataX_test)
y_svm_rbf_lda_train=grid1.predict(DataX_train)
prob_svm_rbf_lda=grid1.predict_proba(DataX_train)
prob_svm_rbf_lda_test=grid1.predict_proba(DataX_test)
metrics(y_test,y_svm_rbf_lda)


# ### RIDGE Y LASSO: ELASTIC NET

# In[129]:


### Clasificador con regularización l1 y l2 por elastic net: regresión logística y SVM
steps = [('ELNET', SGDClassifier(penalty="elasticnet",early_stopping=True,validation_fraction=0.1,random_state=4))]
pipeline = Pipeline(steps)
parametros={'ELNET__loss':['log'],
            'ELNET__l1_ratio':np.linspace(0,1,11)}
grid1=GridSearchCV(pipeline,param_grid=parametros,cv=5,n_jobs=-1,scoring=scoring,refit='kappa')
grid1.fit(DataX_train,y_train)
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
print(grid1.best_params_)
pred_enet=grid1.predict(DataX_test)
pred_enet_train=grid1.predict(DataX_train)
metrics(y_test,pred_enet)
prob_enet=grid1.predict_proba(DataX_train)
prob_enet_test=grid1.predict_proba(DataX_test)


# In[130]:


#revisar cuál fue el modelo que tuvo el mejor comportamiento de los ajustados en ElasticNet
grid1.best_estimator_.get_params()['ELNET']


# ## XGBoost model

# In[131]:


# creamos los pasos para el pipeline dle clasificador

from xgboost import XGBClassifier
steps = [('XGB', XGBClassifier(objective ='binary:hinge', random_state=4))]
pipeline = Pipeline(steps)
parametros={'XGB__n_estimators':[10,15,30,50,100,500,1000, 1500], 'XGB__learning_rate':[0.001, 0.01,0.1,0.5,1]
                , 'XGB__max_depth':[2,3,4,5,8,9,10,15,20]}

grid1 = GridSearchCV(pipeline,param_grid=parametros,cv=5,n_jobs=-1,scoring=scoring,refit='kappa')
grid1.fit(DataX_train,y_train)
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
print(grid1.best_params_)
pred_xgb=grid1.predict(DataX_test)
pred_xgb_train=grid1.predict(DataX_train)
prob_xgb=grid1.predict_proba(DataX_train)
prob_xgb_test=grid1.predict_proba(DataX_test)

xgb = metrics(y_test,pred_xgb)


# ## ADA Boost model

# In[132]:


# creamos los pasos para el pipeline dle clasificador

from sklearn.ensemble import AdaBoostClassifier

steps = [('ADA', AdaBoostClassifier(random_state=4))]
pipeline = Pipeline(steps)
parametros={'ADA__n_estimators':[10, 15, 30, 50, 100, 500, 1000], 'ADA__learning_rate':[0.001, 0.01,0.1,0.5,1]
            ,'ADA__algorithm':['SAMME', 'SAMME.R']}
grid1=GridSearchCV(pipeline,param_grid=parametros,cv=5,n_jobs=-1,scoring=scoring,refit='kappa')
grid1.fit(DataX_train,y_train)
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
print(grid1.best_params_)
pred_ada=grid1.predict(DataX_test)
pred_ada_train=grid1.predict(DataX_train)
prob_ada=grid1.predict_proba(DataX_train)
prob_ada_test=grid1.predict_proba(DataX_test)
ada = metrics(y_test,pred_ada)


# ## Neural networks

# ### MLP

# In[133]:


np.random.seed(1234)
mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=1000)
mlp.fit(DataX_train,y_train)

y_pred = mlp.predict(DataX_test)
y_pred_train = mlp.predict(DataX_train)
proba_mlp=mlp.predict_proba(DataX_train)
proba_mlp_test=mlp.predict_proba(DataX_test)
mlp = metrics(y_test,y_pred)
print(mlp)


# ## Corridas de modelos TensorFlow

# Ahora vamos a ajustar un modelo de una red neuronal recurrente y un modelo de tipo Autoencoder para comparar las predicciones

# In[134]:


print(DataX_train.shape)
print(DataX_test.shape)


# https://www.kaggle.com/code/darvaron/clasificador-binario/notebook

# In[135]:


from tensorflow.keras.utils import to_categorical
np.random.seed(1234)


nrow,ncol=DataX_train.shape
nrow1,ncol1=DataX_test.shape

X_train2=DataX_train[:,0:ncol] # 10 es el rango hasta la cantidad de columnas en el dataset
X_test2=DataX_test[:,0:ncol] # 10 es el rango hasta la cantidad de columnas en el dataset
print("Dimensión original de la base: ",X_train2.shape)

##Redimensionar para que la red neuronal tome valores de entrenamiento en dimensión (90 días, 1 registro)
X_train2=np.reshape(X_train2,(-1,ncol,1))
X_test2=np.reshape(X_test2,(-1,ncol1,1))
y_train2=to_categorical(y_train,2)
y_test2=to_categorical(y_test,2)
print("Dimensión transformada de la base: ",X_train2.shape)


# In[136]:


print(DataX_test.shape)
print(DataX_train.shape)


# In[137]:


# Creamos una semilla para tener reproducibilidad en los resultados del modelo
########################################################################
import numpy as np
import tensorflow as tf
import random as python_random
import os

seed_value= 0
os.environ["PYTHONHASHSEED"] = str(seed_value)

def reset_seeds():
   np.random.seed(123) 
   python_random.seed(123)
   tf.random.set_seed(1234)

reset_seeds() 
########################################################################

import tensorflow as tf
from tensorflow.keras.layers import LSTM,Dense,Conv1D,TimeDistributed,Dropout,BatchNormalization,Flatten,LeakyReLU
from tensorflow.keras.models import Sequential

mod=Sequential()
## capa convolucional que recorre cada
mod.add(Conv1D(filters=64,input_shape=(X_train2.shape[1],1), kernel_size = ncol, activation="relu")) # kernel size se define como ncol del dataset 
mod.add(BatchNormalization())
mod.add(LeakyReLU())
mod.add(LSTM(100,return_sequences=True))
mod.add(Dropout(0.5))
mod.add(LSTM(100,return_sequences=True))
mod.add(Dropout(0.5))
mod.add(LSTM(100,return_sequences=False))
mod.add(Dropout(0.5))
mod.add(Dense(1,activation="sigmoid"))

opt=tf.keras.optimizers.Adam(
    learning_rate=0.001,
)
cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5,restore_best_weights=True)

mod.compile(loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])

mod.fit(
    X_train2,
    y_train,
    validation_data=(X_test2,y_test),
    epochs = 50,
    batch_size = 30,
    callbacks=[cb]
)


# In[138]:


### Revisemos la estructura del modelo
mod.summary()


# In[139]:


##Predecir valores de prueba y realizar métricas
y_pred_lstm_proba=mod.predict(X_train2)
#y_pred_lstm=np.argmax(y_pred_lstm,axis=1)
y_pred_lstm_proba_test=mod.predict(X_test2)

y_pred_lstm=np.where(y_pred_lstm_proba_test>0.5,1,0)
y_pred_lstm_train=np.where(y_pred_lstm_proba>0.5,1,0)

metrics(y_test,y_pred_lstm)


# In[140]:


from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import auc

# Compute fpr, tpr, thresholds and roc auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred_lstm_proba_test)
roc_auc = roc_auc_score(y_test, y_pred_lstm_proba_test)

# Plot ROC curve

ax,fig=plt.subplots(figsize=(10,8))
ax=plt.plot(fpr, tpr, label='Curva ROC (area = %0.3f)' % roc_auc, color="red");
ax=plt.plot([0, 1], [0, 1], '--',color="black",label="Predicción Random") 
ax=plt.xlim([0.0, 1.0])
ax=plt.ylim([0.0, 1.0])
ax=plt.xlabel('1-Especificidad')
ax=plt.ylabel('Sensibilidad')
ax=plt.title('Curva ROC')
ax=plt.legend(loc="lower right")


# ## Autoencoder

# Ahora corremos el modelo usando autoencoder, los cuales utilizan como base la reduccion de dimensionalidad, similar a lo que hacemos con PCA pero este se hace en una de las capas

# In[141]:


X_train_enc=DataX_train.reshape((-1,ncol,1))
X_test_enc=DataX_test.reshape((-1,ncol,1))

# Creamos una semilla para tener reproducibilidad en los reusltados del modelo
import numpy as np
import tensorflow as tf
import random as python_random
import os

seed_value= 0
os.environ["PYTHONHASHSEED"] = str(seed_value)

def reset_seeds():
   np.random.seed(123) 
   python_random.seed(123)
   tf.random.set_seed(1234)

reset_seeds() 


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2,l1

##
encoder_input = Input(shape=(ncol,1), name='encoder_input')
conv1=Conv1D(filters=32,kernel_size=ncol)(encoder_input) #kernel size = 100
relu_cv1=LeakyReLU()(conv1)
bn_cv1=BatchNormalization()(relu_cv1)
conv2=Conv1D(filters=64,kernel_size=1)(bn_cv1) #kernel size = 3
relu_cv2=LeakyReLU()(conv2)
bn_cv2=BatchNormalization()(relu_cv2)
conv3=Conv1D(filters=32,kernel_size=1)(bn_cv2) #kernel size = 3
relu_cv3=LeakyReLU()(conv3)
bn_cv3=BatchNormalization()(relu_cv3)
flatt_cv3=Flatten()(bn_cv3)
dense1=Dense(2)(flatt_cv3)
relu_dense=LeakyReLU()(dense1)
dense_d1=Dense(100,activation="relu")(relu_dense) # dense(100)
dropdense=Dropout(0.6)(dense_d1)
dense2=Dense(1,activation="sigmoid")(dropdense)
autoencod = Model([encoder_input], [dense2],name="autoencoder")

##fit encoder & decoder 

encoder=Model(encoder_input,relu_dense,name="encoder")
decoder_layer=autoencod.layers[-1] 
encoded_input = Input(shape=(100),name="decoder_input")

decoder= Model(encoded_input,decoder_layer(encoded_input),name="decoder")


print(autoencod.summary())
print(encoder.summary())
print(decoder.summary())


# In[142]:


cb2 = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,restore_best_weights=True) #set the best weights found at scoring on validation set

autoencod.compile(
    metrics=['accuracy'],
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,decay=1e-04)  
)

history = autoencod.fit(X_train_enc, y_train, batch_size=30, epochs=50,callbacks=[cb2],validation_data=(X_test_enc,y_test))


# In[143]:


### Cómo se desempeña el modelo que usa autoencoder para realizar la predicción?
y_autoenc_test=np.where(autoencod.predict(X_test_enc)>0.5,1,0)
metrics(y_test,y_autoenc_test)


# Veamos como se desempeña el encoder para reducir dimensionalidad

# In[144]:


encod_data=pd.DataFrame(np.column_stack([encoder.predict(X_train_enc)[:,0:2],y_train]),columns=['dim1','dim2','fraud'])

ax,f=plt.subplots(figsize=(10,8))
ax=sns.scatterplot(x="dim1",y="dim2",hue="fraud",data=encod_data)


# ### STACKING: 
# Vamos a utilizar los cuatro modelos más precisos para elaborar un modelo final que establezca la probabilidad de fraude usando toda la información disponible. Para ello elaboramos un dataset que contiene las probabilidades de cada modelo y la variable objetivo, tanto para el training como el testing set. Posteriormente construimos un clasificador SVM y uno de MLP con esta información.

# In[145]:


proba_mlp.shape


# In[146]:


print(prob_dt_train[:,1].shape)
print(prob_lda_train[:,1].shape)
print(prob_dt_test[:,1].shape)
print(prob_lda_test[:,1].shape)


# In[147]:


data_stack_train=pd.DataFrame({'prob_DT':prob_dt_train[:,1], 'prob_lda': prob_lda_train[:,1],'prob_mlp': proba_mlp[:,1],'prob_svm': prob_svm_rbf_pca[:,1],'prob_lstm':y_pred_lstm_proba[:,0],'target':y_train})
data_stack_test=pd.DataFrame({'prob_DT':prob_dt_test[:,1], 'prob_lda': prob_lda_test[:,1],'prob_mlp': proba_mlp_test[:,1],'prob_svm': prob_svm_rbf_pca_test[:,1],'prob_lstm':y_pred_lstm_proba_test[:,0],'target':y_test})
data_stack_train.head()
###shuffle data
data_stack_train=data_stack_train.sample(frac=1)


# In[148]:


metrica='accuracy'
steps = [("SVM_stack", SVC(kernel="rbf",probability=True))]
pipeline = Pipeline(steps)
parametros = {'SVM_stack__C':[0.01,0.1,1,10], 'SVM_stack__gamma':[0.05,0.01, 1, 5]}
grid1 = GridSearchCV(pipeline, param_grid=parametros, cv=5, scoring=metrica,n_jobs=-1)
grid1.fit(data_stack_train.iloc[:,0:4], data_stack_train.iloc[:,5])
y_svm_stack=grid1.predict(data_stack_test.iloc[:,0:4])
y_svm_stack_train=grid1.predict(data_stack_train.iloc[:,0:4])
prob_svm_stack_test=grid1.predict_proba(data_stack_test.iloc[:,0:4])
metrics(data_stack_test.iloc[:,5],y_svm_stack)
#metrics(data_stack_train.iloc[:,4],y_svm_stack_train)


# In[149]:


##stacking usando MLP
mlp = MLPClassifier(hidden_layer_sizes=(150,150,150), max_iter=1000)
mlp.fit(data_stack_train.iloc[:,0:4], data_stack_train.iloc[:,5])

y_mlp_stack = mlp.predict(data_stack_test.iloc[:,0:4])
y_mlp_stack_train = mlp.predict(data_stack_train.iloc[:,0:4])

proba_mlp=mlp.predict_proba(data_stack_test.iloc[:,0:4])
proba_mlp_test=mlp.predict_proba(data_stack_train.iloc[:,0:4])
metrics(data_stack_test.iloc[:,5],y_mlp_stack)


# ### Otra manera de Stacking:

# In[150]:


from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std

# get a stacking ensemble of models
def get_stacking():
 # define the base models
 level0 = list()
 level0.append(('lr', LogisticRegression()))
 level0.append(('knn', KNeighborsClassifier()))
 level0.append(('cart', DecisionTreeClassifier()))
 level0.append(('rf', RandomForestClassifier()))
 level0.append(('svm', SVC()))
 level0.append(('xgboost', XGBClassifier()))
 # define meta learner model
 level1 = LogisticRegression()
 # define the stacking ensemble
 model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
 return model
 
# get a list of models to evaluate
def get_models():
 models = dict()
 models['lr'] = LogisticRegression()
 models['knn'] = KNeighborsClassifier()
 models['cart'] = DecisionTreeClassifier()
 models['rf'] = RandomForestClassifier()
 models['svm'] = SVC()
 models['xgboost'] = XGBClassifier()
 models['stacking'] = get_stacking()
 return models
 
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
 cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
 return scores

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
 scores = evaluate_model(model, DataX_train, y_train)
 results.append(scores)
 names.append(name)
 print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))


'''
mlp.fit(DataX_train,y_train)

y_pred = mlp.predict(DataX_test)
y_pred_train = mlp.predict(DataX_train)
proba_mlp=mlp.predict_proba(DataX_train)
proba_mlp_test=mlp.predict_proba(DataX_test)
mlp = metrics(y_test,y_pred)
'''


# ### Resumen Modelos Propuestos
# 
# A continuación se resumen las métricas de los modelos ajustados creando una tabla con los valores de las metricas para cada modelo

# In[151]:


name_models=['Decision Tree', 'Randomd Forest', 'XGBoost', 'ADA','KNN','Regresion Ridge','LDA','QDA','SVM Lineal','SVM RBF','SVM Sigmoid',
                'SVM Lineal PCA','SVM RBF PCA','SVM Sigmoid PCA','SVM RBF LDA',
                'Elastic Net','MLP','LSTM','Stacking']

Accuracy_Train=[accuracy_score(y_train,pred_dt_train),accuracy_score(y_train,pred_rf_train),accuracy_score(y_train,pred_xgb_train), accuracy_score(y_train,pred_ada_train),accuracy_score(y_train,pred_knn_train),
                accuracy_score(y_train,pred_lass_train),accuracy_score(y_train,pred_lda_train),accuracy_score(y_train,pred_qda_train),
                accuracy_score(y_train,y_svm_lin_train),accuracy_score(y_train,y_svm_rbf_train),accuracy_score(y_train,y_svm_sig_train),
                accuracy_score(y_train,y_svm_lin_pca_train),
                accuracy_score(y_train,y_svm_rbf_pca_train),accuracy_score(y_train,y_svm_sig_pca_train),
                accuracy_score(y_train,y_svm_rbf_lda_train),accuracy_score(y_train,pred_enet_train),
                accuracy_score(y_train,y_pred_train),accuracy_score(y_train,y_pred_lstm_train),accuracy_score(data_stack_train.iloc[:,5],y_svm_stack_train)]
        
Accuracy_Test=[accuracy_score(y_test,pred_dt),accuracy_score(y_test,pred_rf),accuracy_score(y_test,pred_xgb),accuracy_score(y_test,pred_ada),accuracy_score(y_test,pred_knn),
                accuracy_score(y_test,pred_lass),accuracy_score(y_test,pred_lda),accuracy_score(y_test,pred_qda),
                accuracy_score(y_test,y_svm_lin),accuracy_score(y_test,y_svm_rbf),accuracy_score(y_test,y_svm_sig),
                accuracy_score(y_test,y_svm_lin_pca),
                accuracy_score(y_test,y_svm_rbf_pca),accuracy_score(y_test,y_svm_sig_pca),
                accuracy_score(y_test,y_svm_rbf_lda),accuracy_score(y_test,pred_enet),
                accuracy_score(y_test,y_pred),accuracy_score(y_test,y_pred_lstm),accuracy_score(data_stack_test.iloc[:,5],y_svm_stack)]

Kappa_Train=[cohen_kappa_score(y_train,pred_dt_train),cohen_kappa_score(y_train,pred_rf_train),cohen_kappa_score(y_train,pred_xgb_train),cohen_kappa_score(y_train,pred_ada_train),cohen_kappa_score(y_train,pred_knn_train),
                cohen_kappa_score(y_train,pred_lass_train),cohen_kappa_score(y_train,pred_lda_train),cohen_kappa_score(y_train,pred_qda_train),
                cohen_kappa_score(y_train,y_svm_lin_train),cohen_kappa_score(y_train,y_svm_rbf_train),cohen_kappa_score(y_train,y_svm_sig_train),
                cohen_kappa_score(y_train,y_svm_lin_pca_train),
                cohen_kappa_score(y_train,y_svm_rbf_pca_train),cohen_kappa_score(y_train,y_svm_sig_pca_train),
                cohen_kappa_score(y_train,y_svm_rbf_lda_train),cohen_kappa_score(y_train,pred_enet_train),
                cohen_kappa_score(y_train,y_pred_train),cohen_kappa_score(y_train,y_pred_lstm_train),cohen_kappa_score(data_stack_train.iloc[:,5],y_svm_stack_train)]

Kappa_Test=[cohen_kappa_score(y_test,pred_dt),cohen_kappa_score(y_test,pred_rf),cohen_kappa_score(y_test,pred_xgb),cohen_kappa_score(y_test,pred_ada),cohen_kappa_score(y_test,pred_knn),
                cohen_kappa_score(y_test,pred_lass),cohen_kappa_score(y_test,pred_lda),cohen_kappa_score(y_test,pred_qda),
                cohen_kappa_score(y_test,y_svm_lin),cohen_kappa_score(y_test,y_svm_rbf),cohen_kappa_score(y_test,y_svm_sig),
                cohen_kappa_score(y_test,y_svm_lin_pca),
                cohen_kappa_score(y_test,y_svm_rbf_pca),cohen_kappa_score(y_test,y_svm_sig_pca),
                cohen_kappa_score(y_test,y_svm_rbf_lda),cohen_kappa_score(y_test,pred_enet),
                cohen_kappa_score(y_test,y_pred),cohen_kappa_score(y_test,y_pred_lstm),cohen_kappa_score(data_stack_test.iloc[:,5],y_svm_stack)]

F1Score_Train=[f1_score(y_train,pred_dt_train),f1_score(y_train,pred_rf_train),f1_score(y_train,pred_xgb_train),f1_score(y_train,pred_ada_train),f1_score(y_train,pred_knn_train),
                f1_score(y_train,pred_lass_train),f1_score(y_train,pred_lda_train),f1_score(y_train,pred_qda_train),
                f1_score(y_train,y_svm_lin_train),f1_score(y_train,y_svm_rbf_train),f1_score(y_train,y_svm_sig_train),
                f1_score(y_train,y_svm_lin_pca_train),
                f1_score(y_train,y_svm_rbf_pca_train),f1_score(y_train,y_svm_sig_pca_train),
                f1_score(y_train,y_svm_rbf_lda_train),f1_score(y_train,pred_enet_train),
                f1_score(y_train,y_pred_train),f1_score(y_train,y_pred_lstm_train),f1_score(data_stack_train.iloc[:,5],y_svm_stack_train)]

F1Score_Test=[f1_score(y_test,pred_dt),f1_score(y_test,pred_rf),f1_score(y_test,pred_xgb),f1_score(y_test,pred_ada),f1_score(y_test,pred_knn),
                f1_score(y_test,pred_lass),f1_score(y_test,pred_lda),f1_score(y_test,pred_qda),
                f1_score(y_test,y_svm_lin),f1_score(y_test,y_svm_rbf),f1_score(y_test,y_svm_sig),
                f1_score(y_test,y_svm_lin_pca),
                f1_score(y_test,y_svm_rbf_pca),f1_score(y_test,y_svm_sig_pca),
                f1_score(y_test,y_svm_rbf_lda),f1_score(y_test,pred_enet),
                f1_score(y_test,y_pred),f1_score(y_test,y_pred_lstm),f1_score(data_stack_test.iloc[:,5],y_svm_stack)]

Recall_Train=[recall_score(y_train,pred_dt_train),recall_score(y_train,pred_rf_train),recall_score(y_train,pred_xgb_train),recall_score(y_train,pred_ada_train),recall_score(y_train,pred_knn_train),
                recall_score(y_train,pred_lass_train),recall_score(y_train,pred_lda_train),recall_score(y_train,pred_qda_train),
                recall_score(y_train,y_svm_lin_train),recall_score(y_train,y_svm_rbf_train),recall_score(y_train,y_svm_sig_train),
                recall_score(y_train,y_svm_lin_pca_train),
                recall_score(y_train,y_svm_rbf_pca_train),recall_score(y_train,y_svm_sig_pca_train),
                recall_score(y_train,y_svm_rbf_lda_train),recall_score(y_train,pred_enet_train),
                recall_score(y_train,y_pred_train),recall_score(y_train,y_pred_lstm_train),recall_score(data_stack_train.iloc[:,5],y_svm_stack_train)]

Recall_Test=[recall_score(y_test,pred_dt),recall_score(y_test,pred_rf),recall_score(y_test,pred_xgb),recall_score(y_test,pred_ada),recall_score(y_test,pred_knn),
                recall_score(y_test,pred_lass),recall_score(y_test,pred_lda),recall_score(y_test,pred_qda),
                recall_score(y_test,y_svm_lin),recall_score(y_test,y_svm_rbf),recall_score(y_test,y_svm_sig),
                recall_score(y_test,y_svm_lin_pca),
                recall_score(y_test,y_svm_rbf_pca),recall_score(y_test,y_svm_sig_pca),
                recall_score(y_test,y_svm_rbf_lda),recall_score(y_test,pred_enet),
                recall_score(y_test,y_pred),recall_score(y_test,y_pred_lstm),recall_score(data_stack_test.iloc[:,5],y_svm_stack)]

Tabla={'NombresModelos':name_models,'Accuracy Train':Accuracy_Train,'Accuracy Test':Accuracy_Test,
             'Kappa Train':Kappa_Train,'Kappa Test':Kappa_Test,
             'F1Score Train':F1Score_Train,'F1Score Test':F1Score_Test,
             'Recall Train':Recall_Train,'Recall Test':Recall_Test}
resultado_modelos = pd.DataFrame(Tabla)
resultado_modelos = resultado_modelos.round(decimals = 3)
resultado_modelos.style.highlight_max(color = 'lightblue', axis = 0)
#df.style.highlight_max(color = 'lightgreen', axis = 0)


# ## Salvado del modelo final

# Dados los resultados generales obtenidos con los modelos, se procede a guardar el modelo de decision tree, el cual ha mostrado el mejor desempeno en las metricas

# In[152]:


steps = [('DT', DecisionTreeClassifier(random_state=4))]
pipeline = Pipeline(steps)
parametros = {'DT__max_depth': range(2,20), "DT__min_samples_split": [2, 10, 20, 30, 40, 45, 50, 100]}
grid1 = GridSearchCV(pipeline, param_grid=parametros, cv = 5, scoring = scoring, n_jobs=-1, refit='kappa')
grid1.fit(DataX_train, y_train)
##correr el pipeline
print("score = %3.4f" %(grid1.score(DataX_test,y_test)))
print(grid1.best_params_)
pred_dt_train=grid1.predict(DataX_train)
pred_dt=grid1.predict(DataX_test)
prob_dt_train=grid1.predict_proba(DataX_train)
prob_dt_test=grid1.predict_proba(DataX_test)
metrics(y_test,pred_dt)


# In[153]:


import joblib
filename = 'final_model.sav' # indicamos el nombre del archivo final

# Ajustamos el modelo con los hiperparametros de mejor performance
model = DecisionTreeClassifier(max_depth = 4, min_samples_split = 45, random_state=4)
# usamos el dump de joblib para crear el modelo
joblib.dump(model, filename)


# In[154]:


# load the model from disk
'''
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)
'''


# In[155]:


import types
def imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            yield val.__name__
list(imports())


# In[157]:


get_ipython().system('jupyter nbconvert --output-dir="./reqs" --to script  nb.ipynb')
get_ipython().system('cd reqs')
get_ipython().system('pipreqs')

