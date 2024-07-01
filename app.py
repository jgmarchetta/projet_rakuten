import streamlit as st
import pandas as pd
import numpy as np
import os
import gdown

@st.cache_data
def load_data_preview(csv_path, nrows=5):
    if not os.path.exists(csv_path):
        url = "https://1drv.ms/u/s!As8Ya4n-7uIMhtIBuxFHFX2wL9pbsg?e=zrrvzj"
        gdown.download(url, csv_path, quiet=False)
    
    try:
        return pd.read_csv(csv_path, nrows=nrows)
    except pd.errors.ParserError:
        st.error("Erreur de parsing lors de la lecture des premières lignes du fichier CSV.")
        return None
    except pd.errors.EmptyDataError:
        st.error("Le fichier CSV est vide ou ne contient que des en-têtes.")
        return None
    except UnicodeDecodeError:
        st.error("Erreur d'encodage lors de la lecture du fichier CSV.")
        return None

# Charger les premières lignes pour prévisualiser le fichier
df_preview = load_data_preview('X_train_update.csv')
if df_preview is not None:
    st.write("Aperçu des premières lignes du fichier CSV :")
    st.dataframe(df_preview)
else:
    st.error("Impossible de prévisualiser les premières lignes du fichier CSV.")

