import streamlit as st
import pandas as pd
import joblib

# https://github.com/dataprofessor/ml-app/blob/main/ml-app.py

# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# https://www.analyticsvidhya.com/blog/2021/08/quick-hacks-to-save-machine-learning-model-using-pickle-and-joblib/
# https://towardsdatascience.com/build-and-deploy-machine-learning-web-app-using-pycaret-and-streamlit-28883a569104


#+-----------------+
# Inicio de pagina
## Expancion paginacion
st.set_page_config(page_title = 'Predictor de Fraude - Vehiculo',
                layout = 'wide')

#+--------------------+
# Model building

def import_model(path):
    loaded_model = joblib.load(path)

    return loaded_model





parse_udf = F.udf(..., T.IntegerType())
class Age:
    ...
    def __eq__(self, other: Column):
        return F.lit(self.age) == parse_udf(other)


import re
import pyspark.sql.functions as F
import pyspark.sql.types as T

def connect_to_pyspark(function):
  def helper(age, other):
    myUdf = F.udf(lambda item_from_other: function(age, item_from_other), T.BooleanType())
    return myUdf(other)
  return helper

class Age:

    def __init__(self, age):
      self.age = 45

    def __parse(self, other):
      return int(''.join(re.findall(r'\d', other)))

    @connect_to_pyspark
    def __eq__(self, other):
        return self.age == self.__parse(other)

ages.withColumn("eq20", Age(20) == df.Age).show()




