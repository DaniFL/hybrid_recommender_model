# SISTEMA DE RECOMENDACIÓN DE FILTRADO COLABORATIVO
# Modelo Seleccionado: KNN Basic (Item-Based)

# --- 1. Importación de Librerías ---
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, dump

print("Librerías importadas.")

# --- 2. Carga y Preparación de los Datos ---
try:
    # Carga tu DataFrame preprocesado de la Fase 1
    df_final_reviews = pd.read_csv(r'modelos_desarrollo\data\final_reviews_dataset.csv') 
    print(f"Dataset original cargado con {len(df_final_reviews)} interacciones.")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'final_reviews_preprocessed.csv'.")
    exit()

# Tomamos una muestra aleatoria para que el entrenamiento sea manejable (limitaciones de memoria)
n_sample = 10_000_000
df_sample = df_final_reviews.sample(n=n_sample, random_state=42)
print(f"Trabajando con una muestra de {len(df_sample)} interacciones.")

# --- 3. Entrenamiento del Modelo Final ---

# Definimos el Reader con la escala de nuestra puntuación implícita
reader = Reader(rating_scale=(-0.5, 1.0))

# Cargamos LA MUESTRA en un Dataset de Surprise
data = Dataset.load_from_df(df_sample[['author_steamid', 'appid', 'implicit_rating']], reader)

# Construimos el set de entrenamiento con TODOS los datos de la muestra
# Esto es crucial para el modelo final, ya que queremos que aprenda de toda la información disponible
print("Construyendo el trainset completo...")
trainset = data.build_full_trainset()
print("Trainset construido.")

# Definimos el modelo KNNBasic con los parámetros óptimos
# 'user_based': False indica que es un modelo Item-Based (calcula similitud entre juegos)
sim_options = {'name': 'cosine', 'user_based': False}
knn_model = KNNBasic(k=40, sim_options=sim_options)

# Entrenamos el modelo
print("Entrenando el modelo KNNBasic final...")
knn_model.fit(trainset)
print("Modelo KNNBasic entrenado con éxito.")

# --- 4. Guardado del Modelo ---
output_path = r'modelos_desarrollo\artefactos\knn_collaborative_model.surprise'
print(f"Guardando el modelo en '{output_path}'...")
dump.dump(output_path, algo=knn_model)

print("\n¡Modelo de filtrado colaborativo finalizado y guardado!")