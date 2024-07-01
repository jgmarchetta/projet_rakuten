from bs4 import BeautifulSoup
import pandas as pd

# Lire le fichier HTML
with open('/mnt/data/fichier.html', 'r', encoding='utf-8') as file:
    soup = BeautifulSoup(file, 'html.parser')

# Trouver toutes les tables dans le fichier HTML
tables = soup.find_all('table')

# Si des tables existent, extraire les données et les convertir en CSV
if tables:
    for index, table in enumerate(tables):
        # Lire la table en utilisant pandas
        df = pd.read_html(str(table))[0]
        
        # Sauvegarder la table dans un fichier CSV
        csv_filename = f'/mnt/data/table_{index + 1}.csv'
        df.to_csv(csv_filename, index=False)
        print(f'Table {index + 1} sauvegardée sous {csv_filename}')
else:
    print("Aucune table trouvée dans le fichier HTML.")
