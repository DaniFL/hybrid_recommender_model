# --- 1. Importación de Librerías ---
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, KNNWithMeans, SVD, NMF
from surprise.model_selection import train_test_split
from surprise import accuracy

print("✅ Librerías importadas.")

# --- 2. Carga y Preparación de los Datos ---
try:
    df_final_reviews = pd.read_csv(r'EDA/final_reviews_dataset.csv')
except FileNotFoundError:
    print("Error: No se encontró el archivo 'final_reviews_preprocessed.csv'.")
    exit()

# Tomamos una muestra para evitar errores de memoria.
n_sample = 5_000_000
df_sample = df_final_reviews.sample(n=n_sample, random_state=42)
print(f"Trabajando con una muestra de {len(df_sample)} interacciones.")

# Definimos el Reader y cargamos los datos
reader = Reader(rating_scale=(-0.5, 1.0))
data = Dataset.load_from_df(df_sample[['author_steamid', 'appid', 'implicit_rating']], reader)

# Dividimos en entrenamiento y prueba
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
print("✅ Datos preparados y divididos.")

# --- 3. Definición y Entrenamiento de los Modelos ---
# Definimos los modelos a comparar, añadiendo KNNBasic
models = {
    "KNN": KNNBasic(k=40, sim_options={'name': 'cosine', 'user_based': False}), # <-- AÑADIDO
    "KNN with Means": KNNWithMeans(k=40, sim_options={'name': 'cosine', 'user_based': False}),
    "SVD": SVD(n_factors=100, n_epochs=20, random_state=42),
    "NMF": NMF(n_factors=100, n_epochs=20, random_state=42)
}

results = {}

# Iteramos, entrenamos y evaluamos cada modelo
for name, model in models.items():
    print(f"\n--- Entrenando el modelo: {name} ---")
    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)
    mae = accuracy.mae(predictions, verbose=True)
    results[name] = {'RMSE': rmse, 'MAE': mae}

# --- 4. Resumen y Comparación de Resultados ---
print("\n" + "="*50)
print("RESUMEN DE RENDIMIENTO DE LOS MODELOS")
print("="*50)

results_df = pd.DataFrame.from_dict(results, orient='index')
print(results_df.sort_values(by='RMSE'))