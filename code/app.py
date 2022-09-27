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


