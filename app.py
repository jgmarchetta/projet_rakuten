import streamlit as st
import pandas as pd
import gdown

def download_file_from_google_drive(drive_url, output_path):
    gdown.download(drive_url, output_path, quiet=False)

# URL du fichier Google Drive (lien de partage)
drive_url = 'https://drive.google.com/uc?id=1dI7zqHcU1XhafXHNyk_339Oh6iPvqAzO'

# Chemin de sortie pour le fichier téléchargé
output_path = 'X_train_update.csv'

# Télécharger le fichier
download_file_from_google_drive(drive_url, output_path)

# Lire le fichier CSV
df = pd.read_csv(output_path)

# Afficher le dataframe dans Streamlit
st.write("Aperçu du fichier CSV téléchargé :")
st.dataframe(df)
