import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from tqdm import tqdm
import numpy as np
import time

# --- 1. Carga y Preparación de Datos ---
print("Cargando datos...")
df_games = pd.read_csv(r'EDA\steam_games_final.csv', dtype={14: str})
df_reviews = pd.read_csv(r'EDA\final_reviews_dataset.csv')

# Usamos la misma muestra que en el entrenamiento colaborativo para mayor consistencia
n_sample = 5_000_000
df_sample = df_reviews.sample(n=n_sample, random_state=42)

# --- 2. Construcción fuente de verdad (Matriz de Co-ocurrencia) ---
print("Construyendo matriz de Ground Truth desde el comportamiento de usuarios...")
# Creamos una matriz dispersa usuario-juego (1 si le gustó, 0 si no)
liked_reviews = df_sample[df_sample['implicit_rating'] > 0]

# Mapeamos los IDs a índices numéricos consecutivos
user_c = pd.Categorical(liked_reviews['author_steamid'])
item_c = pd.Categorical(liked_reviews['appid'])
user_map = {uid: i for i, uid in enumerate(user_c.categories)}
item_map = {iid: i for i, iid in enumerate(item_c.categories)}
reverse_item_map = {i: iid for iid, i in item_map.items()}

# Creamos la matriz dispersa
user_item_matrix = csr_matrix((np.ones(len(liked_reviews)), 
                               (user_c.codes, item_c.codes)), 
                              shape=(len(user_c.categories), len(item_c.categories)))

# Calculamos la co-ocurrencia: game x game
co_occurrence_matrix = user_item_matrix.T.dot(user_item_matrix)
# La similitud de un juego consigo mismo no nos interesa
co_occurrence_matrix.setdiag(0) 
print("✅ Matriz de Ground Truth construida.")

# --- 3. Entrenamiento de los Modelos de Contenido ---
print("▶️  Entrenando modelos de contenido...")
df_games['text_features'] = (df_games['name'].fillna('') + ' ' + df_games['short_description'].fillna('') + ' ' + df_games['genres'].fillna('').replace(';', ' ') + ' ' + df_games['categories'].fillna('').replace(';', ' '))
df_games.reset_index(inplace=True, drop=True)
name_to_idx = pd.Series(df_games.index, index=df_games['name']).drop_duplicates()
idx_to_appid = pd.Series(df_games.appid.values, index=df_games.index)

# Modelo A: TF-IDF + KNN
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=20000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_games['text_features'])
knn_tfidf = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute')
knn_tfidf.fit(tfidf_matrix)

def get_tfidf_recs(idx, k=10):
    query_vector = tfidf_matrix[idx]
    distances, indices = knn_tfidf.kneighbors(query_vector)
    return idx_to_appid.iloc[indices.flatten()[1:]].tolist()

# Modelo B: LSA + KNN
svd_model = TruncatedSVD(n_components=300, random_state=42)
latent_matrix = svd_model.fit_transform(tfidf_matrix)
knn_lsa = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute')
knn_lsa.fit(latent_matrix)

def get_lsa_recs(idx, k=10):
    query_vector = latent_matrix[idx].reshape(1, -1)
    distances, indices = knn_lsa.kneighbors(query_vector)
    return idx_to_appid.iloc[indices.flatten()[1:]].tolist()

print("Modelos de contenido entrenados.")

# --- 4. Bucle de Evaluación ---
print("Iniciando evaluación empírica...")
k = 10
tfidf_precisions = []
lsa_precisions = []

# Tomamos una muestra de juegos para evaluar
test_games_appids = np.random.choice(list(item_map.keys()), size=1000, replace=False)

for appid in tqdm(test_games_appids):
    # Obtenemos la fuente de verdad para este juego
    try:
        item_idx_mapped = item_map[appid]
        ground_truth_scores = co_occurrence_matrix[item_idx_mapped].toarray().flatten()
        top_ground_truth_indices = np.argsort(ground_truth_scores)[-k:]
        ground_truth_appids = {reverse_item_map[i] for i in top_ground_truth_indices}
    except (KeyError, IndexError):
        continue

    # Obtenemos las recomendaciones de nuestros modelos
    try:
        game_row = df_games[df_games['appid'] == appid].iloc[0]
        game_main_idx = game_row.name # El índice principal del DataFrame
    except IndexError:
        continue
        
    tfidf_recs = set(get_tfidf_recs(game_main_idx, k=k))
    lsa_recs = set(get_lsa_recs(game_main_idx, k=k))
    
    # Calculamos la precisión para cada modelo
    tfidf_hits = len(tfidf_recs & ground_truth_appids)
    lsa_hits = len(lsa_recs & ground_truth_appids)
    
    tfidf_precisions.append(tfidf_hits / k)
    lsa_precisions.append(lsa_hits / k)

# --- 5. Resultados Finales ---
avg_tfidf_precision = np.mean(tfidf_precisions) if tfidf_precisions else 0
avg_lsa_precision = np.mean(lsa_precisions) if lsa_precisions else 0

print("\n" + "="*50)
print("RESULTADOS DE LA EVALUACIÓN EMPÍRICA")
print("="*50)
print(f"Precision@10 (TF-IDF + KNN): {avg_tfidf_precision:.2%}")
print(f"Precision@10 (LSA + KNN):   {avg_lsa_precision:.2%}")