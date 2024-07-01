import streamlit as st

@st.cache_data
def read_file_head(csv_path, nlines=5):
    if not os.path.exists(csv_path):
        url = "https://1drv.ms/u/s!As8Ya4n-7uIMhtIBuxFHFX2wL9pbsg?e=zrrvzj"
        gdown.download(url, csv_path, quiet=False)
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            head_lines = []
            for _ in range(nlines):
                head_lines.append(file.readline())
        return head_lines
    except UnicodeDecodeError:
        st.error("Erreur d'encodage lors de la lecture du fichier CSV.")
        return None

# Afficher les premières lignes brutes du fichier CSV
raw_head = read_file_head('X_train_update.csv')
if raw_head is not None:
    st.write("Premières lignes brutes du fichier CSV :")
    st.text("".join(raw_head))
else:
    st.error("Impossible de lire les premières lignes brutes du fichier CSV.")
