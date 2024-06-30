import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import gdown

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Embedding, Concatenate, GlobalAveragePooling2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
except ImportError as e:
    st.error(f"Erreur lors de l'importation de TensorFlow: {e}")
    st.stop()

import torch
import plotly.graph_objs as go

from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, Dataset, SequentialSampler
from streamlit_elements import elements, mui

# Définir la configuration de la page pour utiliser toute la largeur de l'écran
st.set_page_config(layout="wide")

# Définir le CSS directement dans le script
css = """
<style>
body, h1, h2, h3, h4, h5, h6, p, div, span, li, a, input, button, .stText, .stMarkdown, .stSidebar, .stTitle, .stHeader, .stRadio {
    font-family: 'Arial', sans-serif;
}
.css-1d391kg, .css-18e3th9, .css-1l02zno, .css-1v3fvcr, .css-qbe2hs, .css-1y4p8pa {
    font-family: 'Arial', sans-serif;
}
.small-title {
    font-size: 14px;  /* Ajustez cette valeur selon vos besoins */
    font-weight: bold;
}
.reduced-spacing p {
    margin-bottom: 5px;  /* Ajustez cette valeur selon vos besoins */
    margin-top: 5px;     /* Ajustez cette valeur selon vos besoins */
}
.red-title {
    color: #BF0000;
}
.stRadio > div > div {
    background-color: #BF0000 !important;
}
.center-title {
    text-align: center;
}
</style>
"""

# Injecter le CSS dans l'application Streamlit
st.markdown(css, unsafe_allow_html=True)

# st.sidebar.title("PROJET")

# Ajouter le logo de Rakuten dans la barre latérale avec un lien hypertexte
st.sidebar.markdown(f"""
<a href="https://challengedata.ens.fr/participants/challenges/35/" target="_blank">
    <img src='https://fr.shopping.rakuten.com/visuels/0_content_square/autres/rakuten-logo6.svg' style="width: 100%;">
</a>
""", unsafe_allow_html=True)

st.sidebar.title("Sommaire")
pages = ["Présentation", "Données", "Pré-processing", "Machine Learning", "Deep Learning", "Conclusion", "Démo"]
page = st.sidebar.radio("Aller vers:", pages)

@st.cache_data
def load_data(csv_path):
    if not os.path.exists(csv_path):
        url = "https://1drv.ms/u/s!As8Ya4n-7uIMhtIBuxFHFX2wL9pbsg?e=zrrvzj"
        gdown.download(url, csv_path, quiet=False)
    return pd.read_csv(csv_path)

@st.cache_data
def load_model_and_tokenizer(model_url, tokenizer_path, le_path):
    # Télécharger le modèle depuis Google Drive
    model_path = "model_EfficientNetB0-LSTM.keras"
    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False)
    
    model = load_model(model_path)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(le_path, 'rb') as handle:
        le = pickle.load(handle)
    return model, tokenizer, le

@st.cache_data
def preprocess_data(df, _label_encoder):
    df.loc[:, "prdtypecode"] = _label_encoder.fit_transform(df["prdtypecode"])
    train_texts, val_texts, train_labels, val_labels = train_test_split(df["token_text"], df["prdtypecode"], test_size=0.2, random_state=42)
    return train_texts, val_texts, train_labels, val_labels

def tokenize_texts(_tokenizer, texts, max_len=128):
    encodings = _tokenizer(list(texts), truncation=True, padding=True, return_tensors="pt", max_length=max_len)
    return encodings

def prepare_datasets(texts, _labels, _tokenizer, max_len=128):
    class CustomDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.long)
            }

    dataset = CustomDataset(texts=texts.tolist(), labels=_labels.tolist(), tokenizer=_tokenizer, max_len=max_len)
    return dataset

def predict_and_evaluate(_model, loader, device):
    _model.eval()
    predictions = []
    true_labels = []
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in loader:
        input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)

        with torch.no_grad():
            outputs = _model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        logits = outputs.logits

        total_eval_loss += loss.item()
        preds = torch.argmax(logits, dim=1).flatten()
        accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        total_eval_accuracy += accuracy
        nb_eval_steps += 1

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    avg_val_accuracy = total_eval_accuracy / nb_eval_steps
    avg_val_loss = total_eval_loss / nb_eval_steps

    return np.array(predictions), np.array(true_labels), avg_val_accuracy, avg_val_loss

# Présentation du projet
if page == "Présentation":
    st.markdown("<h1 class='red-title center-title'>Projet Rakuten - Classification Multimodal</h1>", unsafe_allow_html=True)

    # Tabs for different sections
    tab1, tab2 = st.tabs(["Contexte", "Objectif du projet"])

    with tab1:
        st.write("""
        Dans le cadre d'un challenge organisé par l'ENS et de notre formation data scientist au sein de DataScientest, nous avons pu travailler sur la classification de produits à grande échelle, en développant un projet visant à prédire le type de chaque produit tel que défini dans le catalogue de Rakuten France.
        """)

        col1, col2, col3 = st.columns(3)
        with col2:
            st.image('rakuten_image_entreprise.jpg', caption='Siège social de Rakuten à Futakotamagawa, Tokyo', width=550, use_column_width=True)
        
        st.write("""
        **Rakuten, Inc. (Rakuten Kabushiki-gaisha)**  
        Société japonaise de services internet créée en février 1997. Depuis juin 2010, Rakuten a acquis PriceMinister, premier site de commerce électronique en France.
        """)

    with tab2:
        st.write("""
        **Objectif du projet**  
        L’objectif du projet est la classification multimodale à grande échelle des données de produits en codes de types de produits.  
        Il s’agit alors de prédire le code type des produits à partir de données textuelles et images.
        """)

        st.image('objectif_projet.png', caption='Objectif de projet', width=1000, use_column_width=False)

# Exploration des données
if page == "Données":
    st.markdown("<h1 class='red-title center-title'>Exploration des données</h1>", unsafe_allow_html=True)
    st.write('Le projet comporte 3 jeux de données textuelles et 1 jeu de donnée images.')

    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("- X_train\n- X_test\n- Y_train\n- Fichier Images scindé en 2 fichiers image_train & image_test")
        st.markdown("</div>", unsafe_allow_html=True)

    # Ajouter un selectbox pour choisir le jeu de données
    selected_dataset = st.selectbox("**Sélectionnez le jeu de données :**", ["X_train", "X_test", "Y_train", "Fichier Images Train"])

    # Charger les données
    df_train = load_data('X_train_update.csv')
    df_test = load_data('X_test_update.csv')
    df_target = load_data('Y_train_CVw08PX.csv')

    # Fonctions pour afficher des graphiques
    def plot_missing_values_heatmap(df):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(df.isnull(), cmap=sns.color_palette(['#828282', '#BF0000']), cbar=False, ax=ax)
        st.pyplot(fig)

    def plot_nan_percentage(df, column_name):
        nan_percentage = (df[column_name].isnull().sum() / len(df)) * 100
        non_nan_percentage = 100 - nan_percentage
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(["Valeurs non-NaN", "NaN"], [non_nan_percentage, nan_percentage], color=["#004BAA", "#BF0000"])
        ax.set_xlim(0, 100)
        for i, v in enumerate([non_nan_percentage, nan_percentage]):
            ax.text(v + 1, i, f"{v:.2f}%", color="black", va="center")
        st.pyplot(fig)

    def plot_duplicate_percentage(df, column_name):
        duplicate_counts = df[column_name].duplicated().value_counts()
        unique_percentage = (duplicate_counts[False] / len(df)) * 100
        duplicate_percentage = (duplicate_counts[True] / len(df)) * 100
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(["Valeurs uniques", "Doublons"], [unique_percentage, duplicate_percentage], color=["#004BAA", "#BF0000"])
        ax.set_xlim(0, 100)
        for i, v in enumerate([unique_percentage, duplicate_percentage]):
            ax.text(v + 1, i, f"{v:.2f}%", color="black", va="center")
        st.pyplot(fig)
        
    # Afficher les informations en fonction du choix
    if selected_dataset == "X_train":
        st.write("Vous avez sélectionné le jeu de données X_train.")
        st.data_editor(
            df_train.head(),
            column_config={
                "productid": st.column_config.NumberColumn(format="%d"),
                "imageid": st.column_config.NumberColumn(format="%d")
            },
            hide_index=True,
        )
        st.image('image_variable X_train_file.png', caption='Schéma des variables du jeu de données X_train', width=1000, use_column_width=False)
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        with col1:
            plot_missing_values_heatmap(df_train)
        with col2:
            plot_nan_percentage(df_train, column_name='description')
        with col3:
            plot_duplicate_percentage(df_train, column_name='designation')
        with col4:
            st.image('histogramme_langues_X_train.png', width=100, use_column_width=True)
            
    elif selected_dataset == "X_test":
        st.write("Vous avez sélectionné le jeu de données X_test.")
        st.data_editor(
            df_test.head(),
            column_config={
                "productid": st.column_config.NumberColumn(format="%d"),
                "imageid": st.column_config.NumberColumn(format="%d")
            },
            hide_index=True,
        )
        st.image('image variable X_test_file.png', caption='Schéma des variables du jeu de données X_test', width=1000, use_column_width=False)
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        with col1:
            plot_missing_values_heatmap(df_test)
        with col2:
            plot_nan_percentage(df_test, column_name='description')
        with col3:
            plot_duplicate_percentage(df_test, column_name='designation')
        with col4:
            st.image('histogramme_langues_X_test.png', width=100, use_column_width=True)

    elif selected_dataset == "Y_train":
        st.write("Vous avez sélectionné le jeu de données Y_train.")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df_target.head())
        with col2:
            st.image('image_variable Y_train_file.png', caption='Schéma des variables du jeu de données Y_train', width=1000, use_column_width=True)
        col3, col4 = st.columns(2)
        with col3:
            plot_missing_values_heatmap(df_target)
        with col4:
            st.image('visualisation_pdt_categorie.png', caption='Visualisation du nombre de produit par catégorie', width=100, use_column_width=True)
        st.image('categorie_produit.png', caption='27 Catégories de produits', width=500, use_column_width=False)

    elif selected_dataset == "Fichier Images Train":
        st.write("Vous avez sélectionné le fichier d'images du jeu de données images.")
        st.write("Le jeu de données images comporte 2 fichiers : 1 image_train et 1 image_test.")
        col1, col2 = st.columns(2)
        with col1:
            st.image('visualisation_fichier_image.png', caption='Visualisation du fichier images_train', width=100, use_column_width=True)
        with col2:
            st.image('dataframe_images_train.png', caption='DataFrame du fichier images_train', width=100, use_column_width=True)
        show_rapprochement = st.checkbox('**Rapprochement Textes-Images-Cible**')
        if show_rapprochement:
            st.markdown("<h2 style='text-align: center; color: #004BAA; font-family: Calibri, sans-serif;'>Rapprochement Textes - Images - Variable Cible</h2>", unsafe_allow_html=True)
            st.write('Voici un exemple une visualisation de rapprochement de textes, des images avec la variable cible correspondante')
            st.image('rapprochement texte_image_cible.png', caption='rapprochement texte/image & catégorie de produit', width=1000, use_column_width=False)

# prétraitement
if page == "Pré-processing":
    st.markdown("<h1 class='red-title center-title'>Pré-processing</h1>", unsafe_allow_html=True)
    
    st.write("""
    Pour ce projet, nous avons utilisé 2 catégories de données non structurées : texte et image.
    Ces données peuvent être vectorisées de plusieurs manières différentes. Nous avons donc essayé plusieurs scénarios pour trouver le plus performant en testant les modèles.
    Voici les différents scénarios de pré-processing testés.
    """)

    st.image('scenario_preprocessing.png', caption='Représentation des scénarios de pré-processing', width=1000, use_column_width=False)

    show_scenarios = st.checkbox('**Afficher les scénarios**')
    if show_scenarios:
        st.markdown('**Scénario A :** Vectorisation des images par CNN, vectorisation du texte avec SPACY sans traduction de texte')
        st.markdown('**Scénario B :** Vectorisation des images par CNN, vectorisation du texte avec TF-IDF, après tokenisation, lemmatisation, application des stop-words, sans traduction de texte et une réduction par TruncatedSVD')
        st.markdown('**Scénario C :** Même vectorisation que le scénario B, sans application de réduction')
        st.markdown('**Scénario D :** Vectorisation des images par une succession de transformations des images (transformations en gris, application de filtre Gaussien puis Laplacian, et réduction de la taille des images), vectorisation du texte avec TF-IDF, après tokenisation, lemmatisation, application des stop-words, sans traduction de texte et une réduction par TruncatedSVD')
        st.markdown('**Scénario E :** Même vectorisation que scénario B avec traduction de texte dans la langue majoritaire, à savoir le français')

elif page == "Machine Learning":
    st.markdown("<h1 class='red-title center-title'>Machine Learning</h1>", unsafe_allow_html=True)
    
    # Définir les onglets
    tabs = st.tabs(["Scénario A", "Scénario B", "Scénario E", "Amélioration", "Optimisation"])

    # Fonction pour ajouter les expanders de modèles avec des images spécifiques
    def add_model_expanders(images):
        with st.expander(f"**XGboost** Score F1-pondéré: {images['XGboost']['score']}"):
            st.write(f"Détails sur le modèle XGboost.")
            st.image(images['XGboost']['path'], caption=None, width=1200)
        with st.expander(f"**SGD Classifier** Score F1-pondéré: {images['SGD Classifier']['score']}"):
            st.write(f"Détails sur le modèle SGD Classifier.")
            st.image(images['SGD Classifier']['path'], caption=None, width=1200)
        with st.expander(f"**Random Forest** Score F1-pondéré: {images['Random Forest']['score']}"):
            st.write(f"Détails sur le modèle Random Forest.")
            st.image(images['Random Forest']['path'], caption=None, width=1200)
        with st.expander(f"**Voting Classifier 'Soft'** Score F1-pondéré: {images['Voting Classifier Soft']['score']}"):
            st.write(f"Détails sur le modèle Voting Classifier 'Soft'.")
            st.image(images['Voting Classifier Soft']['path'], caption=None, width=1200)
        with st.expander(f"**Voting Classifier 'Hard'** Score F1-pondéré: {images['Voting Classifier Hard']['score']}"):
            st.write(f"Détails sur le modèle Voting Classifier 'Hard'.")
            st.image(images['Voting Classifier Hard']['path'], caption=None, width=1200)
        with st.expander(f"**Naive Bayes Gaussien** Score F1-pondéré: {images['Naive Bayes Gaussien']['score']}"):
            st.write(f"Détails sur le modèle Naive Bayes Gaussien.")
            st.image(images['Naive Bayes Gaussien']['path'], caption=None, width=1200)

    # Images pour chaque scénario avec scores
    images_scenario_A = {
        'XGboost': {'path': 'A_XGboost.png', 'score': 0.73},
        'SGD Classifier': {'path': 'A_SGD Classifier.png', 'score': 0.68},
        'Random Forest': {'path': 'A_Random Forest.png', 'score': 0.65},
        'Voting Classifier Soft': {'path': 'A_VCS.png', 'score': 0.73},
        'Voting Classifier Hard': {'path': 'A_VCH.png', 'score': 0.72},
        'Naive Bayes Gaussien': {'path': 'A_NBG.png', 'score': 0.46}
    }

    images_scenario_B = {
        'XGboost': {'path': 'B_XGboost.png', 'score': 0.77},
        'SGD Classifier': {'path': 'B_SGD Classifier.png', 'score': 0.62},
        'Random Forest': {'path': 'B_Random Forest.png', 'score': 0.76},
        'Voting Classifier Soft': {'path': 'B_VCS.png', 'score': 0.77},
        'Voting Classifier Hard': {'path': 'B_VCH.png', 'score': 0.76},
        'Naive Bayes Gaussien': {'path': 'B_NBG.png', 'score': 0.51}
    }

    images_scenario_E = {
        'XGboost': {'path': 'E_XGboost.png', 'score': 0.76},
        'SGD Classifier': {'path': 'E_SGD Classifier.png', 'score': 0.58},
        'Random Forest': {'path': 'E_Random Forest.png', 'score': 0.75},
        'Voting Classifier Soft': {'path': 'E_VCS.png', 'score': 0.75},
        'Voting Classifier Hard': {'path': 'E_VCH.png', 'score': 0.74},
        'Naive Bayes Gaussien': {'path': 'E_NBG.png', 'score': 0.50}
    }

    # Contenu du tab Scénario A
    with tabs[0]:
        st.header("Scénario A")
        st.write("Vectorisation des images par CNN, vectorisation du texte avec SPACY sans traduction de texte")
        st.write("")
        st.write("Les modèles:")
        add_model_expanders(images_scenario_A)

    # Contenu du tab Scénario B
    with tabs[1]:
        st.header("Scénario B")
        st.write("Vectorisation des images par CNN, vectorisation du texte avec TF-IDF, après tokenisation, lemmatisation, application des stop-words, sans traduction de texte et une réduction par TruncatedSVD")
        st.write("")
        st.write("Les modèles:")
        add_model_expanders(images_scenario_B)

    # Contenu du tab Scénario E
    with tabs[2]:
        st.header("Scénario E")
        st.write("Même vectorisation que scénario B avec traduction de texte dans la langue majoritaire, à savoir le français")
        st.write("")
        st.write("Les modèles:")
        add_model_expanders(images_scenario_E)

    # Contenu du tab Amélioration
    with tabs[3]:
        st.header("Amélioration")
        st.write("Étape 1 : Recherche des Meilleurs Hyperparamètres")
        st.write("")
        with st.expander("**Amélioration B** Score F1-pondéré: 0.79"):
            st.write("Détails sur Amélioration B.")
            st.image("ameb.png", caption=None, width=400)

    # Contenu du tab Optimisation
    with tabs[4]:
        st.header("Optimisation")
        st.write("Étape 2 : Validation Croisée")
        st.write("Étape 3 : Rééchantillonnage et Évaluation")
        st.write("")
        with st.expander("**SMOTE** Score F1-pondéré: 0.80"):
            st.write("Détails sur SMOTE.")
            st.image("SMOTE.png", caption=None, width=400)
        with st.expander("**RandomUnderSampler** Score F1-pondéré: 0.74"):
            st.write("Détails sur RandomUnderSampler.")
            st.image("RUS.png", caption=None, width=400)

elif page == "Deep Learning":
    st.markdown("<h1 class='red-title center-title'>Les modèles de Deep Learning étudiés</h1>", unsafe_allow_html=True)
 
    # Tab creation
    tab1, tab2, tab3, tab4 = st.tabs(["Benchmark", "Les modèles étudiés", "Synthèse des scores de F1-pondéré", "Interprétation"])

    with tab1:
        st.header("Description du Benchmark - Challenge Rakuten")
        st.write("""
        Tout en gardant notre fil de conduite, nous avons réfléchi à plusieurs stratégies de modélisation Deep Learning.  
        Dans un premier temps, nous avons essayé de reproduire les modèles proposés par le « Challenge » afin d’atteindre le score du Benchmark.  
        (Il a été notre référence minimum de score de F1-pondéré)         
                 
        L'algorithme du challenge utilise deux modèles distincts pour les images et le texte.

        **Pour les données d'image :**
        - Il utilise une version du modèle Residual Networks (ResNet50) de Keras. 
        - Le modèle est pré-entraîné avec un ensemble de données ImageNet.
        
        **Pour les données textuelles :**
        - Il utilise un classificateur CNN simplifié.
        - La taille d'entrée est la longueur de désignation maximale possible, 34 dans ce cas.
        - L'architecture se compose d'une couche d'intégration et de 6 blocs convolutionnels à pooling maximal.
        
        **Performances de référence**  
        Voici le score F1 pondéré obtenu à l'aide des 2 modèles du Benchmark décrits ci-dessus :
        - Images : **0,5534**
        - Texte : **0,8113**

        Comme le modèle utilisant du texte est plus performant, Rakuten a utilisé celui-ci comme référence de son Benchmark.
        """)

    with tab2:
        st.header("Les modèles")
        st.write("**Scénario B**")
        st.write("Vectorisation des images par CNN, vectorisation du texte avec TF-IDF, après tokenisation, lemmatisation, application des stop-words, sans traduction de texte et une réduction par TruncatedSVD")

        with st.expander("**Réseau de neurones denses avec Keras** Score F1-pondéré: 0.77"):
            st.write("""
            Les DNN sont une architecture de réseau neuronal artificiel couramment utilisée pour la classification et la prédiction.    
            
            **Description de l'Architecture du Modèle DNN**
            **4 couches** entièrement connectées (ou couches denses).

            - **Couche Dense 1** : La première couche dense prend une entrée et la transforme en une sortie de 512 neurones, avec 113 152 paramètres entraînables.
            - **Couche Dense 2** : La deuxième couche dense réduit cette sortie à 256 neurones, avec 131 328 paramètres entraînables.
            - **Couche Dense 3** : La troisième couche dense réduit encore la sortie à 128 neurones, avec 32 896 paramètres entraînables.
            - **Couche Dense 4** : La dernière couche dense produit la sortie finale de 27 neurones, avec 3 483 paramètres entraînables.

            **Nombre Total de Paramètres**
            - Total des paramètres entraînables : 280 859
            - Total des paramètres non-entraînables : 0 

            **Avantages :**
            - Capacité d'apprentissage des relations complexes données textuelles et images
            - Flexibilité de l’architecture 
            - Robuste aux variations
            """)

            st.write("")
            st.write("")

            # Charger les images
            image1a = "rep_dnn.png"
            image2a = "mtx_dnn.png"

            # Créer deux colonnes
            col1, col2 = st.columns(2)

            # Afficher la première image dans la première colonne
            with col1:
                st.image(image1a, caption='Rapport de classification - Réseau de neurones denses avec Keras', width=450)

            # Afficher la deuxième image dans la deuxième colonne
            with col2:
                st.image(image2a, caption='Matrice de confusion - Réseau de neurones denses avec Keras')

        with st.expander("**ResNet50 - LSTM** Score F1-pondéré: 0.74"):
            st.write("""
            Ce modèle hybride combinant:  
            ResNet50 est une architecture de réseau neuronal convolutif (CNN) performante pour l'extraction de caractéristiques visuelles à partir d'images.  
            LSTM (Long Short-Term Memory) est un type de réseau neuronal récurrent (RNN) spécialisé dans le traitement de données séquentielles, telles que le texte.  
            
            **Description de l'Architecture du Modèle Hybride ResNet50-LSTM**  
            **12 couches**

            - **input_layer (InputLayer)** : (None, 100) - Entrée pour les données textuelles.
            - **input_layer_1 (InputLayer)** : (None, 128, 128, 3) - Entrée pour les images.
            - **embedding (Embedding)** : (None, 100, 128) - Transforme les données textuelles en représentations denses de 128 dimensions.
            - **resnet50 (Functional)** : (None, 4, 4, 2048) - Réseau ResNet50 pré-entraîné pour l'extraction de caractéristiques visuelles.
            - **lstm (LSTM)** : (None, 64) - Réseau LSTM pour capturer les dépendances temporelles dans les données textuelles.
            - **global_average_pooling2d (GlobalAveragePooling2D)** : (None, 2048) - Réduction de la sortie de ResNet50.
            - **dense (Dense)** : (None, 64) - Couche dense appliquée à la sortie du LSTM.
            - **dense_1 (Dense)** : (None, 128) - Couche dense appliquée à la sortie du GlobalAveragePooling.
            - **concatenate (Concatenate)** : (None, 192) - Concatenation des sorties denses.
            - **dense_2 (Dense)** : (None, 256) - Couche dense pour fusionner les représentations.
            - **dropout (Dropout)** : (None, 256) - Couche de régularisation pour éviter le surapprentissage.
            - **dense_3 (Dense)** : (None, 27) - Couche de sortie finale pour la classification.

            **Nombre Total de Paramètres**
            - Total des paramètres : 25,239,899 (96.28 MB)
            - Paramètres entraînables : 1,652,187 (6.30 MB)
            - Paramètres non-entraînables : 23,587,712 (89.98 MB)  

            **Avantages :**
            - Exploitation efficace des informations multimodales : ResNet50 - images, LSTM - texte
            - Apprentissage des relations temporelles : LSTM - compréhension du sens des phrases 
            - Classification précise : Combinaison ResNet50-LSTM classification précise et robuste
            """)

            st.write("")
            st.write("")

            # Charger les images
            image1b = "rep_dnn.png"
            image2b = "mtx_dnn.png"

            # Créer deux colonnes
            col1, col2 = st.columns(2)

            # Afficher la première image dans la première colonne
            with col1:
                st.image(image1b, caption='Rapport de classification - ResNet50 - LSTM', width=450)

            # Afficher la deuxième image dans la deuxième colonne
            with col2:
                st.image(image2b, caption='Matrice de confusion - ResNet50 - LSTM')    
            
        with st.expander("**DistilBERT** Score F1-pondéré: 0.92"):
            st.write("""
            DistilBERT est une version allégée du modèle BERT, qui est un modèle de langage pré-entraîné révolutionnaire pour le traitement du langage naturel (NLP).  
            Il est spécialement conçu pour des tâches de classification, de compréhension et de génération de texte.
            
            **Description Brève de l'Architecture du Modèle DistilBERT**  
            88 couches

            - **distilbert.embeddings.word_embeddings.weight** : (30522, 768) - 23,440,896 paramètres
            - **distilbert.embeddings.position_embeddings.weight** : (512, 768) - 393,216 paramètres
            - **distilbert.embeddings.LayerNorm.weight** : (768,) - 768 paramètres
            - **distilbert.embeddings.LayerNorm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.attention.q_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.0.attention.q_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.attention.k_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.0.attention.k_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.attention.v_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.0.attention.v_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.attention.out_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.0.attention.out_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.sa_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.sa_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.ffn.lin1.weight** : (3072, 768) - 2,359,296 paramètres
            - **distilbert.transformer.layer.0.ffn.lin1.bias** : (3072,) - 3,072 paramètres
            - **distilbert.transformer.layer.0.ffn.lin2.weight** : (768, 3072) - 2,359,296 paramètres
            - **distilbert.transformer.layer.0.ffn.lin2.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.output_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.output_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.attention.q_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.1.attention.q_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.attention.k_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.1.attention.k_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.attention.v_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.1.attention.v_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.attention.out_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.1.attention.out_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.sa_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.sa_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.ffn.lin1.weight** : (3072, 768) - 2,359,296 paramètres
            - **distilbert.transformer.layer.1.ffn.lin1.bias** : (3072,) - 3,072 paramètres
            - **distilbert.transformer.layer.1.ffn.lin2.weight** : (768, 3072) - 2,359,296 paramètres
            - **distilbert.transformer.layer.1.ffn.lin2.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.output_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.output_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.attention.q_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.2.attention.q_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.attention.k_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.2.attention.k_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.attention.v_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.2.attention.v_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.attention.out_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.2.attention.out_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.sa_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.sa_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.ffn.lin1.weight** : (3072, 768) - 2,359,296 paramètres
            - **distilbert.transformer.layer.2.ffn.lin1.bias** : (3072,) - 3,072 paramètres
            - **distilbert.transformer.layer.2.ffn.lin2.weight** : (768, 3072) - 2,359,296 paramètres
            - **distilbert.transformer.layer.2.ffn.lin2.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.output_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.output_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.attention.q_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.3.attention.q_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.attention.k_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.3.attention.k_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.attention.v_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.3.attention.v_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.attention.out_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.3.attention.out_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.sa_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.sa_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.ffn.lin1.weight** : (3072, 768) - 2,359,296 paramètres
            - **distilbert.transformer.layer.3.ffn.lin1.bias** : (3072,) - 3,072 paramètres
            - **distilbert.transformer.layer.3.ffn.lin2.weight** : (768, 3072) - 2,359,296 paramètres
            - **distilbert.transformer.layer.3.ffn.lin2.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.output_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.output_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.attention.q_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.4.attention.q_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.attention.k_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.4.attention.k_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.attention.v_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.4.attention.v_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.attention.out_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.4.attention.out_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.sa_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.sa_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.ffn.lin1.weight** : (3072, 768) - 2,359,296 paramètres
            - **distilbert.transformer.layer.4.ffn.lin1.bias** : (3072,) - 3,072 paramètres
            - **distilbert.transformer.layer.4.ffn.lin2.weight** : (768, 3072) - 2,359,296 paramètres
            - **distilbert.transformer.layer.4.ffn.lin2.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.output_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.output_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.attention.q_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.5.attention.q_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.attention.k_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.5.attention.k_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.attention.v_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.5.attention.v_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.attention.out_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.5.attention.out_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.sa_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.sa_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.ffn.lin1.weight** : (3072, 768) - 2,359,296 paramètres
            - **distilbert.transformer.layer.5.ffn.lin1.bias** : (3072,) - 3,072 paramètres
            - **distilbert.transformer.layer.5.ffn.lin2.weight** : (768, 3072) - 2,359,296 paramètres
            - **distilbert.transformer.layer.5.ffn.lin2.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.output_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.output_layer_norm.bias** : (768,) - 768 paramètres
            - **pre_classifier.weight** : (768, 768) - 589,824 paramètres
            - **pre_classifier.bias** : (768,) - 768 paramètres
            - **classifier.weight** : (27, 768) - 20,736 paramètres
            - **classifier.bias** : (27,) - 27 paramètres

            **Nombre Total de Paramètres**
            - Total des paramètres : 66,974,235

            **Avantages :**
            - Excellente capacité à capturer les nuances et la sémantique des données textuelles
            - Réduit les coûts de calcul et d'entraînement grâce à sa taille plus petite
            - Favorise une utilisation efficace des ressources matérielles et temps de traitement plus rapides
            - Adaptabilité élevée à différentes tâches NLP grâce à son architecture flexible et pré-entraînée
            """)
                     
            st.write("")
            st.write("")
            
            # Charger les images
            image1c = "rep_d_bert.png"
            image2c = "mtx_d_bert.png"

            # Créer deux colonnes
            col1, col2 = st.columns(2)

            # Afficher la première image dans la première colonne
            with col1:
                st.image(image1c, caption='Rapport de classification - DistilBERT', width=450)

            # Afficher la deuxième image dans la deuxième colonne
            with col2:
                st.image(image2c, caption='Matrice de confusion - DistilBERT')                   
                  
        with st.expander("**EfficientNetB0 - LSTM** Score F1-pondéré: 0.96"):
            st.write("""
            Ce modèle hybride combinant:
            EfficientNetB0 est une famille de modèles de réseaux neuronaux convolutifs (CNN) qui se distinguent par leur efficacité en termes de précision et de coût computationnel pour le traitement d'images.   
            LSTM, quant à lui, est un réseau neuronal récurrent (RNN) spécialisé dans le traitement de données séquentielles, telles que le texte. 
            
            **Description de l'Architecture du Modèle Hybride EfficientNetB0-LSTM**  
            **12 couches**

            - **input_layer (InputLayer)** : (None, 100) - Entrée pour les données textuelles.
            - **input_layer_1 (InputLayer)** : (None, 128, 128, 3) - Entrée pour les images.
            - **embedding (Embedding)** : (None, 100, 128) - Transforme les données textuelles en représentations denses de 128 dimensions.
            - **efficientnetb0 (Functional)** : (None, 4, 4, 1280) - Réseau EfficientNetB0 pré-entraîné pour l'extraction de caractéristiques visuelles.
            - **lstm (LSTM)** : (None, 128) - Réseau LSTM pour capturer les dépendances temporelles dans les données textuelles.
            - **global_average_pooling2d (GlobalAveragePooling2D)** : (None, 1280) - Réduction de la sortie de EfficientNetB0.
            - **dense (Dense)** : (None, 128) - Couche dense appliquée à la sortie du LSTM.
            - **dense_1 (Dense)** : (None, 128) - Couche dense appliquée à la sortie du GlobalAveragePooling.
            - **concatenate (Concatenate)** : (None, 256) - Concatenation des sorties denses.
            - **dense_2 (Dense)** : (None, 256) - Couche dense pour fusionner les représentations.
            - **dropout (Dropout)** : (None, 256) - Couche de régularisation pour éviter le surapprentissage.
            - **dense_3 (Dense)** : (None, 27) - Couche de sortie finale pour la classification.

            **Nombre Total de Paramètres**
            - Total des paramètres : 5,714,366 (21.80 MB)
            - Paramètres entraînables : 1,664,795 (6.35 MB)
            - Paramètres non-entraînables : 4,049,571 (15.45 MB)

            **Avantages :**
            - Excellente capacité à extraire des caractéristiques pertinentes à partir d'images avec une efficacité computationnelle élevée
            - Favorise une utilisation judicieuse des ressources matérielles grâce à son architecture efficace
            - Permet une compréhension profonde des données séquentielles, telles que les descriptions textuelles, grâce à LSTM
            - Offre une classification précise et robuste en combinant les forces de l'extraction d'informations visuelles d'EfficientNetB0 et de la compréhension du sens des phrases de LSTM
            """)
            
            st.write("")
            st.write("")
            
            # Charger les images
            image1d = "rep_eff_lstm.png"
            image2d = "mtx_eff_LSTM.png"

            # Créer deux colonnes
            col1, col2 = st.columns(2)

            # Afficher la première image dans la première colonne
            with col1:
                st.image(image1d, caption='Rapport de classification - EfficientNetB0 - LSTM', width=450)

            # Afficher la deuxième image dans la deuxième colonne
            with col2:
                st.image(image2d, caption='Matrice de confusion - EfficientNetB0 - LSTM')       
           
        with st.expander("**Modèles Non-Aboutis**"):
            st.write("""
            Il a été testé d’autres stratégies de modèle de Deep Learning qui n’ont pas aboutis principalement pour des raisons techniques, de matériel ou de surcoût.

            **Un modèle hybride profond combinant VGG16 et BERT**
            - Le modèle se compose de deux branches principales : 
            - Une Branche VGG16 : Cette branche traite les images d'entrée. Elle utilise l'architecture VGG16 pré-entraînée pour extraire des caractéristiques visuelles de haut niveau à partir des images.
            - Une Branche BERT : Cette branche traite les descriptions textuelles d'entrée. Elle utilise le modèle BERT pré-entraîné pour extraire des représentations sémantiques des textes.
            - Les sorties des deux branches sont ensuite fusionnées pour obtenir une représentation combinée qui capture à la fois des informations visuelles et textuelles. Cette représentation combinée est ensuite utilisée pour effectuer la tâche finale.

            **Le modèle VisualBert**
            - Il permet de comprendre et de traiter conjointement du texte et des images. Il s'appuie sur l'architecture Transformer et utilise deux mécanismes d'apprentissage préliminaire innovants pour capturer des relations sémantiques entre les mots et les régions d'une image.
            - Son fonctionnement est le suivant : 
            - Intégration des informations : VisualBert traite d'abord le texte et l'image séparément à l'aide de deux encodeurs Transformer.
            - Attention mutuelle : Ensuite, il utilise un mécanisme d'attention mutuelle pour permettre aux encodeurs de s'influencer mutuellement et d'aligner les mots du texte avec les régions pertinentes de l'image.
            - Représentation finale : Les sorties des deux encodeurs sont ensuite combinées pour obtenir une représentation finale qui capture à la fois les informations textuelles et visuelles.
            - Ainsi, malgré une compréhension approfondie du modèle et de son architecture, les limitations matérielles ont empêché une exécution complète et efficace de VisualBERT sur ma machine.
        
            **Un modèle multimodal**
            - Capable de traiter et de comprendre des informations provenant de plusieurs sources sensorielles, telles que la vision, l'ouïe et le langage. Son fonctionnement est le suivant :
            - Intégration des données : Le modèle reçoit des données provenant de différentes modalités, comme des images, des sons ou du texte.
            - Extraction de caractéristiques : Il extrait ensuite des caractéristiques pertinentes de chaque modalité de données.
            - Fusion des caractéristiques : Ces caractéristiques sont ensuite fusionnées en une représentation commune qui capture les informations essentielles de toutes les modalités.
            - Apprentissage et prédiction : Le modèle utilise cette représentation commune pour apprendre une tâche spécifique, comme la classification d'objets, la reconnaissance vocale ou la traduction automatique.
            - Limitation des Ressources : Bien que ce modèle détecte les images, il connaît le même problème que VisualBERT. L'exécution du modèle sur ma machine entraîne des arrêts dus à l'insuffisance de ressources matérielles.
            """)        

    with tab3:
        st.header("Synthèse des scores de F1-pondéré")
        st.write("""
                Voici un récapitulatif des scores des modèles étudiés qui ont fonctionnés.  
                """)
        
        col1, col2 = st.columns(2)

        with col1:
            st.write("""
                    Le modèle EfficientNetB0-LSTM se distingue comme le meilleur modèle parmi ceux testés, avec une exactitude, une précision, un rappel et un score F1 pondéré les plus élevés (0.96).   
                    Avec ce modèle nous avons dépassé non seulement le score du Benchmark (0.81), mais aussi le meilleur score du « Challenge » (0.92).  
                    A noté que le modèle DistilBERT atteint lui aussi le niveau du meilleur score, sans le dépasser.  
                    Si l’on analyse séparément les scores texte et image, l’on constate que le modèle LSTM a probablement contribué à atteindre ce score.  
                    Mais ce résultat, provient aussi de l’étape de pré-processing effectuée sur le texte en filtrant un maximum d’anomalies et en préparant le texte à la modélisation.  

                    En revanche, le modèle ResNet50-LSTM affiche les performances les plus faibles, notamment avec un score F1 pondéré (0.74), ce qui est significativement inférieur aux autres modèles.   
                    Ces résultats suggèrent que les modèles basés sur des réseaux de neurones convolutifs sont particulièrement efficaces pour cette tâche de classification, tandis que les DNN pourraient nécessiter des améliorations ou ne pas être aussi bien adaptés à ce type spécifique de données ou de problème.

                    **Nous avons donc choisi le modèle EfficientNetB0-LSTM pour passer à l'étape de prédictions de nos données.**  
                    """)
            
        with col2:       
            st.image("score_deep.png", caption="Synthèse des scores de F1-pondéré")   

    with tab4:
        st.header("Interprétation du texte et des images par le Modèle EfficientNetB0-LSTM")
        st.write("""
        Pour une meilleure compréhension des résultats, nous avons réalisé quelques visualisations du comportement de notre modèle EfficientNetB0-LSTM sur des exemples de descriptions de produits et d'images.
        """)
        
        with st.expander("**Interprétation des descriptions textuelles de produit**"):
            st.write("""
            Visualisation des Activations Intermédiaires :
            - Un moyen puissant d'observer ce qu'un réseau de neurones a appris à différentes étapes est d'inspecter ses activations intermédiaires. Pour ce faire, nous avons utilisé des échantillons de descriptions de produits et examiné les sorties des couches cachées de notre modèle EfficientNetB0-LSTM. 
            - Cela nous a permis de comprendre comment chaque couche transformait les données textuelles et quelles caractéristiques étaient extraites à différents niveaux de profondeur.

            Représentations Vectorielles :
            - Nous avons également converti les descriptions textuelles en représentations vectorielles et visualisé ces vecteurs dans un espace de caractéristiques réduit (à l'aide de techniques de réduction de dimensionnalité telles que t-SNE ou PCA). 
            - Cette visualisation nous a aidé à identifier comment les descriptions similaires étaient groupées et séparées dans l'espace de caractéristiques appris par le modèle.

            **Exemples de descriptions analysées :**

            Description 1 :
            - "Nouvelle chemise en coton pour homme, manches longues, disponible en plusieurs tailles et couleurs, parfaite pour toutes les occasions."
            - La visualisation des activations a montré une forte réponse dans les couches associées à la reconnaissance des vêtements, mettant en évidence des mots clés comme "chemise", "homme", "coton", "tailles" et "couleurs".

            Description 2 :
            - "Chaussures de sport légères pour femmes, conçues pour le jogging, avec une semelle antidérapante et un excellent soutien de la voûte plantaire."
            - Les activations intermédiaires ont révélé une attention particulière aux termes "chaussures", "sport", "femmes", "jogging", "semelle antidérapante", et "soutien de la voûte plantaire", indiquant que le modèle était capable de comprendre le contexte et les caractéristiques spécifiques du produit.

            Description 3 :
            - "Montre-bracelet élégante avec un cadran en acier inoxydable et un bracelet en cuir véritable, résistante à l'eau jusqu'à 50 mètres."
            - La représentation vectorielle a montré que cette description était bien séparée des autres catégories, avec une activation élevée autour des mots "montre-bracelet", "acier inoxydable", "bracelet en cuir", et "résistante à l'eau".
            """)

            st.write("")
            st.write("")

            # Charger les images
            image3a = "rep_efficient.png"

            # Afficher les images
            st.image(image3a, caption='Interprétation des descriptions textuelles de produit', width=800)
            
        with st.expander("**Interprétation des images de produit**"):
            st.write("""
            Activation des Couches Convolutives :
            - Pour mieux comprendre comment notre modèle EfficientNetB0-LSTM traite les images de produits, nous avons visualisé les activations des couches convolutives. 
            - Ces visualisations ont révélé quelles parties de l'image le modèle trouvait les plus importantes pour prendre sa décision de classification.
            - Par exemple, pour une image de chaussure de sport, les couches convolutives ont montré des activations élevées autour de la forme de la chaussure et des détails de la semelle, indiquant que le modèle apprenait à identifier des caractéristiques visuelles spécifiques pertinentes pour la catégorie de produit.

            Cartes de Classe :
            - Nous avons également généré des cartes de classe pour certaines images, montrant quelles régions de l'image contribuaient le plus à la prédiction de la classe de produit. 
            - Ces cartes de classe ont permis de visualiser la "zone d'attention" du modèle sur les images, confirmant que le modèle se concentrait sur les aspects visuellement significatifs des produits.
            - Pour une montre-bracelet, par exemple, la carte de classe a montré une attention particulière sur le cadran et le bracelet, des éléments cruciaux pour l'identification de ce type de produit.

            **Exemples d'images analysées :**

            Image 1 :
            - Une image de chemise en coton pour homme.
            - Les activations ont montré une forte réponse autour du col, des boutons et des manches de la chemise, des caractéristiques distinctives de ce type de vêtement.

            Image 2 :
            - Une image de chaussures de sport pour femmes.
            - Les couches convolutives ont fortement activé les contours de la chaussure et les motifs de la semelle, mettant en évidence les éléments clés utilisés par le modèle pour classer cette image.

            Image 3 :
            - Une image de montre-bracelet élégante.
            - La carte de classe a révélé que le modèle se concentrait principalement sur le cadran et le bracelet de la montre, confirmant que ces régions contenaient les informations essentielles pour la classification.
            """)

            st.write("")
            st.write("")

            # Charger les images
            image4a = "cam_efficient.png"

            # Afficher les images
            st.image(image4a, caption='Interprétation des images de produit', width=800)

elif page == "Conclusion":
    st.markdown("<h1 class='red-title center-title'>Conclusion</h1>", unsafe_allow_html=True)

    st.write("""
    Le projet de classification multimodale pour les produits Rakuten a été un voyage riche en apprentissage et en innovation. Nous avons exploré et comparé diverses approches, allant des modèles de Machine Learning traditionnels aux architectures de Deep Learning les plus avancées. Voici quelques points clés de notre travail :

    1. **Exploration et Prétraitement des Données :**
    - Nous avons soigneusement exploré les jeux de données textuels et visuels, identifié les défis liés aux données manquantes et aux doublons, et appliqué diverses techniques de prétraitement pour préparer les données pour la modélisation.

    2. **Modélisation Machine Learning :**
    - Nous avons testé plusieurs scénarios de modélisation, chacun avec des approches différentes pour vectoriser les textes et les images. Les modèles XGboost, SGD Classifier, Random Forest, et les Voting Classifiers ont montré des performances variées, mais c'est l'approche basée sur TF-IDF et la réduction de dimensionnalité qui s'est révélée la plus prometteuse.

    3. **Modélisation Deep Learning :**
    - En explorant les modèles de Deep Learning, nous avons constaté que les architectures combinant des réseaux de neurones convolutifs (CNN) pour les images et des modèles LSTM pour les textes offraient des performances exceptionnelles. 
    - Le modèle EfficientNetB0-LSTM a surpassé les attentes avec un score F1 pondéré de 0.96, établissant une nouvelle référence pour la tâche de classification des produits.

    4. **Interprétation et Visualisation :**
    - Pour garantir que nos modèles ne soient pas des "boîtes noires", nous avons investi du temps dans l'interprétation des activations intermédiaires et des cartes de classe, offrant ainsi une vue transparente sur le fonctionnement interne de nos modèles et leur prise de décision.

    5. **Défis et Limitations :**
    - Bien que nous ayons rencontré des limitations techniques et matérielles, en particulier avec certains modèles de Deep Learning avancés comme VisualBERT, nous avons su adapter nos stratégies et tirer parti des architectures les plus efficaces disponibles.

    **Perspectives Futures :**
    - Pour aller plus loin, nous pourrions envisager d'explorer des architectures de modèle encore plus sophistiquées, telles que les modèles multimodaux non aboutis. 
    - L'intégration d'une analyse plus approfondie des activations intermédiaires et des représentations vectorielles pourrait également offrir des pistes pour affiner encore davantage nos modèles.

    En conclusion, notre projet a démontré que les techniques avancées de Machine Learning et de Deep Learning peuvent considérablement améliorer la classification des produits dans des environnements commerciaux à grande échelle comme Rakuten. Nous sommes convaincus que les approches développées ici peuvent servir de base solide pour des applications futures et contribuer à l'optimisation des processus de classification dans le secteur du commerce électronique.
    """)

elif page == "Démo":
    st.markdown("<h1 class='red-title center-title'>Démo</h1>", unsafe_allow_html=True)
    st.markdown("### Prédiction du type de produit à partir de sa description et de son image")

    # Exemple de description
    description = st.text_area("Entrez la description du produit:", value="Chaussures de sport légères pour femmes, conçues pour le jogging, avec une semelle antidérapante et un excellent soutien de la voûte plantaire.")
    
    # Exemple d'image
    image = st.file_uploader("Téléchargez une image du produit:", type=["jpg", "jpeg", "png"])

    # Bouton de prédiction
    if st.button("Prédire"):
        if description and image:
            # Afficher les entrées de l'utilisateur
            st.write("**Description du produit:**")
            st.write(description)

            st.write("**Image du produit:**")
            st.image(image, caption="Image téléchargée", use_column_width=True)

            # Ajouter un message d'état de traitement
            st.write("**Prédiction en cours...**")

            # Chargement du modèle et du tokenizer
            model_url = "https://drive.google.com/uc?id=1-rGSvPR5Ng5dJqaw85a93SOZWG2L9SUu"
            tokenizer_path = "tokenizer_distilbert.pickle"
            le_path = "label_encoder.pickle"
            model, tokenizer, label_encoder = load_model_and_tokenizer(model_url, tokenizer_path, le_path)

            # Prétraitement de la description
            description_tokenized = tokenize_texts(tokenizer, [description])

            # Prétraitement de l'image
            image_pil = Image.open(image)
            image_resized = image_pil.resize((128, 128))
            image_array = img_to_array(image_resized) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Prédiction
            predictions = model.predict([description_tokenized["input_ids"], description_tokenized["attention_mask"], image_array])
            predicted_label = label_encoder.inverse_transform(np.argmax(predictions, axis=1))[0]

            # Afficher la prédiction
            st.write("**Type de produit prédit:**")
            st.write(predicted_label)
        else:
            st.write("Veuillez entrer une description et télécharger une image pour obtenir une prédiction.")

