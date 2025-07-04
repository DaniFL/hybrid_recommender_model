# recommender_final.py
import pandas as pd
import joblib
from surprise import dump
import numpy as np
import os

def obtener_recomendaciones_hibridas(user_id, game_title, k=10):
    """
    Función principal que carga los modelos y genera una recomendación híbrida.
    Maneja el caso de arranque en frío para usuarios nuevos.
    """
    # --- 1. Carga de Artefactos y Datos ---
    print("Cargando modelos y datos...")
    
    # Lista de archivos de modelo necesarios
    required_files = {
        'preprocessor': r'modelos_desarrollo\artefactos\content_preprocessor.joblib',
        'svd': r'modelos_desarrollo\artefactos\content_svd_model.joblib',
        'knn_content': r'modelos_desarrollo\artefactos\content_knn_model.joblib',
        'knn_collab': r'modelos_desarrollo\artefactos\knn_collaborative_model.surprise',
        'games_df': r'modelos_desarrollo\data\steam_games_final.csv',
        'reviews_df': r'modelos_desarrollo\data\final_reviews_dataset.csv'
    }

    # Verificación de que todos los archivos existen
    for key, path in required_files.items():
        if not os.path.exists(path):
            return f"Error: No se encontró el archivo de modelo necesario: '{path}'. Asegúrate de haber entrenado todos los modelos."

    # Carga de los modelos y datos
    preprocessor = joblib.load(required_files['preprocessor'])
    svd_model = joblib.load(required_files['svd'])
    knn_content_model = joblib.load(required_files['knn_content'])
    _, knn_collab_model = dump.load(required_files['knn_collab'])
    df_games = pd.read_csv(required_files['games_df'], dtype={14: str})
    df_reviews = pd.read_csv(required_files['reviews_df'])
    
    print("Todos los artefactos cargados con éxito.")

    # --- 2. Preparación de Datos Auxiliares ---
    # Replicamos la ingeniería de características para el modelo de contenido
    df_games['required_age'] = pd.to_numeric(df_games['required_age'], errors='coerce').fillna(0)
    df_games['developers_main'] = df_games['developers'].str.split(',|;').str[0].str.strip().fillna('unknown')
    df_games['text_features'] = (df_games['name'].fillna('') + ' ' + df_games['short_description'].fillna('') + ' ' + df_games['genres'].fillna('').replace(';', ' ') + ' ' + df_games['categories'].fillna('').replace(';', ' '))
    
    # Recreamos la matriz latente
    X_processed = preprocessor.transform(df_games)
    X_latent = svd_model.transform(X_processed)
    
    # Creamos mapeos de nombre a índice
    df_games.reset_index(inplace=True, drop=True)
    name_to_idx = pd.Series(df_games.index, index=df_games['name']).drop_duplicates()

    # --- 3. Generación de Recomendaciones ---
    
    # a) Recomendaciones de Contenido
    try:
        idx = name_to_idx[game_title]
        query_vector = X_latent[idx].reshape(1, -1)
        distances, indices = knn_content_model.kneighbors(query_vector, n_neighbors=k+1)
        content_recs_indices = indices.flatten()[1:]
        content_recs = list(df_games['name'].iloc[content_recs_indices])
    except KeyError:
        return f"Error: El juego '{game_title}' no se encontró en el catálogo."
        
    # b) Lógica de Arranque en Frío y Recomendaciones Colaborativas
    known_users = set(df_reviews['author_steamid'].unique())
    if user_id not in known_users:
        print(f"AVISO: Usuario {user_id} no reconocido (arranque en frío). Devolviendo solo recomendaciones de contenido.")
        return content_recs

    # Si el usuario es conocido, generamos recomendaciones colaborativas
    user_played_appids = set(df_reviews[df_reviews['author_steamid'] == user_id]['appid'])
    all_games_appids = df_games['appid'].unique()
    games_to_predict_appids = np.setdiff1d(list(all_games_appids), list(user_played_appids))
    
    predictions = [knn_collab_model.predict(user_id, game_id) for game_id in games_to_predict_appids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    collab_recs_appids = [pred.iid for pred in predictions[:k]]
    collab_recs = list(df_games[df_games['appid'].isin(collab_recs_appids)]['name'])

    # c) Hibridación por Intercalado (Mixed Hybridization)
    hybrid_list = []
    for i in range(k):
        if i < len(content_recs):
            hybrid_list.append(content_recs[i])
        if i < len(collab_recs):
            hybrid_list.append(collab_recs[i])
            
    # Eliminamos duplicados manteniendo el orden y tomamos los k primeros
    final_recs = list(pd.Series(hybrid_list).drop_duplicates().head(k))
    return final_recs


# --- PUNTO DE ENTRADA PRINCIPAL PARA PRUEBAS ---
if __name__ == '__main__':
    # --- PRUEBA 1: USUARIO EXISTENTE ---
    USER_EXISTENTE = 76561198055526826 
    JUEGO_REFERENCIA = 'The Witcher 3: Wild Hunt'
    
    print("\n" + "="*50)
    print("PRUEBA 1: USUARIO CONOCIDO")
    print("="*50)
    print(f"Usuario: {USER_EXISTENTE}")
    print(f"Juego de referencia: '{JUEGO_REFERENCIA}'")
    
    recomendaciones1 = obtener_recomendaciones_hibridas(USER_EXISTENTE, JUEGO_REFERENCIA)
    
    print("\n--- Recomendaciones Híbridas ---")
    if isinstance(recomendaciones1, list) and recomendaciones1:
        for i, game in enumerate(recomendaciones1, 1):
            print(f"{i}. {game}")
    else:
        print(recomendaciones1)

    # --- PRUEBA 2: USUARIO NUEVO (NO EXISTENTE) ---
    USER_NUEVO = 99999999999999999
    
    print("\n" + "="*50)
    print("PRUEBA 2: USUARIO NUEVO (ARRANQUE EN FRÍO)")
    print("="*50)
    print(f"Usuario: {USER_NUEVO}")
    print(f"Juego de referencia: '{JUEGO_REFERENCIA}'")
    
    recomendaciones2 = obtener_recomendaciones_hibridas(USER_NUEVO, JUEGO_REFERENCIA)
    
    print("\n--- Recomendaciones de Contenido (Fallback) ---")
    if isinstance(recomendaciones2, list) and recomendaciones2:
        for i, game in enumerate(recomendaciones2, 1):
            print(f"{i}. {game}")
    else:
        print(recomendaciones2)