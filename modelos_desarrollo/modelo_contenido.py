# MODELO DE CONTENIDO FINAL
# Metodología: LSA (TF-IDF + SVD) + KNN

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import joblib
import time

print("Iniciando entrenamiento del modelo de contenido final...")
start_time = time.time()

# --- 1. Carga y Preparación de Datos ---
try:
    df_games = pd.read_csv(r'modelos_desarrollo\data\steam_games_final.csv', dtype={14: str})
    print(f"Dataset cargado con {len(df_games)} juegos.")
except FileNotFoundError:
    print("Error: No se encontró 'steam_games_final.csv'.")
    exit()

# Ingeniería de características
df_games['required_age'] = pd.to_numeric(df_games['required_age'], errors='coerce').fillna(0)
df_games['developers_main'] = df_games['developers'].str.split(',|;').str[0].str.strip().fillna('unknown')
df_games['text_features'] = (df_games['name'].fillna('') + ' ' + 
                             df_games['short_description'].fillna('') + ' ' + 
                             df_games['genres'].fillna('').replace(';', ' ') + ' ' +
                             df_games['categories'].fillna('').replace(';', ' '))
print("Ingeniería de características completada.")


# --- 2. Entrenamiento de los Componentes ---
# a) Preprocesador (TF-IDF, OneHotEncoder, etc.)
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=20000), 'text_features'),
        ('categorical', OneHotEncoder(handle_unknown='ignore', min_frequency=10), ['developers_main']),
        ('numeric', StandardScaler(), ['required_age'])
    ],
    remainder='drop'
)
print("Ajustando el preprocesador...")
X_processed = preprocessor.fit_transform(df_games)
print("Preprocesador ajustado.")

# b) Modelo SVD (para LSA)
svd_model = TruncatedSVD(n_components=300, random_state=42)
print("Ajustando el modelo SVD...")
X_latent = svd_model.fit_transform(X_processed)
print(f"SVD ajustado. Matriz latente creada con forma: {X_latent.shape}")

# c) Modelo KNN (para búsqueda de vecinos)
knn_model = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute')
print("Ajustando el modelo KNN...")
knn_model.fit(X_latent)
print("Modelo KNN ajustado.")


# --- 3. Guardado de los Artefactos Finales ---
joblib.dump(preprocessor, r'modelos_desarrollo\artefactos\content_preprocessor.joblib')
joblib.dump(svd_model, r'modelos_desarrollo\artefactos\content_svd_model.joblib')
joblib.dump(knn_model, r'modelos_desarrollo\artefactos\content_knn_model.joblib')

end_time = time.time()
print("\n" + "="*50)
print("¡Entrenamiento del modelo de contenido completado!")
print(f"Tiempo total: {end_time - start_time:.2f} segundos.")
print("Se han guardado los siguientes archivos:")
print("   - content_preprocessor.joblib")
print("   - content_svd_model.joblib")
print("   - content_knn_model.joblib")
print("="*50)