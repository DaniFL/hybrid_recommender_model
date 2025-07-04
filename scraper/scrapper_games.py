import pandas as pd
import asyncio
import aiohttp
import os
import sys
import requests
from tqdm.asyncio import tqdm_asyncio
import csv
import time

# --- CONFIGURACIÓN HÍBRIDA Y ROBUSTA ---
OUTPUT_CSV_PATH = 'steam_games_dataset_final.csv'
PROXY_FILE_PATH = 'proxies.txt'

# Cada proxy esperará 1.6s entre sus propias peticiones. 300s / 1.6s = 187.5 peticiones en 5 min. (Por debajo del límite de 200)
REQUEST_INTERVAL_PER_PROXY = 1.6
# La concurrencia ahora puede ser igual al número de proxies. Cada uno trabajará a su ritmo.
CONCURRENT_REQUESTS = 100 # Ajústalo al número de proxies que tienes (ej. 100)
MAX_PROXY_FAILURES = 5 # Si un proxy falla 5 veces, se descarta.

# --- URLs ---
APP_LIST_URL = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
APP_DETAILS_URL = "https://store.steampowered.com/api/appdetails"

# --- Clase para gestionar el rate limit de cada proxy ---
class ProxyRateLimiter:
    def __init__(self, proxy_url, interval):
        self.proxy_url = proxy_url
        self.interval = interval
        self.last_request_time = 0
        self.failures = 0

    async def wait_for_permission(self):
        """Espera el tiempo necesario para cumplir el rate limit."""
        now = time.monotonic()
        time_since_last = now - self.last_request_time
        if time_since_last < self.interval:
            await asyncio.sleep(self.interval - time_since_last)
        self.last_request_time = time.monotonic()
    
    def record_failure(self):
        self.failures += 1
        return self.failures >= MAX_PROXY_FAILURES

def load_proxies():
    # ... (Función sin cambios)
    if not os.path.exists(PROXY_FILE_PATH):
        print(f"Error: No se encuentra el archivo de proxies '{PROXY_FILE_PATH}'.")
        return None
    with open(PROXY_FILE_PATH, 'r') as f:
        proxies = [line.strip() for line in f if line.strip()]
    if not proxies:
        print("Error: El archivo de proxies está vacío.")
        return None
    print(f"Cargados {len(proxies)} proxies desde '{PROXY_FILE_PATH}'.")
    return proxies

def fetch_all_app_ids():
    # ... (Función sin cambios)
    print("-> Iniciando Paso 1: Obteniendo la lista completa de AppIDs de Steam...")
    try:
        response = requests.get(APP_LIST_URL, timeout=60)
        response.raise_for_status()
        data = response.json()
        apps = data.get("applist", {}).get("apps", [])
        if not apps:
            print("Error: No se encontraron aplicaciones en la respuesta de la API.")
            return []
        print(f"-> Paso 1 completado: Se encontraron {len(apps)} aplicaciones en total.")
        return [app['appid'] for app in apps]
    except requests.exceptions.RequestException as e:
        print(f"Error de red en el Paso 1: {e}", file=sys.stderr)
        return None

async def save_row_to_csv(game_data, file_lock):
    # ... (Función sin cambios)
    async with file_lock:
        file_exists = os.path.exists(OUTPUT_CSV_PATH) and os.path.getsize(OUTPUT_CSV_PATH) > 0
        with open(OUTPUT_CSV_PATH, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=game_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(game_data)


async def worker(worker_id, app_id_queue, proxy_limiters, file_lock, session, pbar):
    """Un 'worker' que coge appids y proxies de una cola."""
    while not app_id_queue.empty():
        try:
            app_id, proxy_url = await app_id_queue.get()
            limiter = proxy_limiters[proxy_url]
            
            await limiter.wait_for_permission()
            
            # --- MODIFICADO: Se elimina el filtro de idioma 'l=spanish' ---
            params = {'appids': app_id}
            
            response = await session.get(APP_DETAILS_URL, params=params, proxy=proxy_url, timeout=45)
            
            if response.status == 200:
                data = await response.json(content_type=None)
                if data and str(app_id) in data and data[str(app_id)].get('success'):
                    app_data = data[str(app_id)]['data']
                    if app_data.get('type') == 'game':
                        # Lógica de extracción...
                        categories_list = [cat.get('description') for cat in app_data.get('categories', []) if cat.get('description')]
                        genres_list = [gen.get('description') for gen in app_data.get('genres', []) if gen.get('description')]
                        platforms_list = [platform for platform, available in app_data.get('platforms', {}).items() if available]
                        game_details = {
                            'appid': app_data.get('steam_appid'), 'type': app_data.get('type'),
                            'name': app_data.get('name'), 
                            'short_description': app_data.get('short_description'),
                            'categories': ', '.join(categories_list), 
                            'genres': ', '.join(genres_list),
                            'developers': ', '.join(app_data.get('developers', [])),
                            'publishers': ', '.join(app_data.get('publishers', [])),
                            'release_date': app_data.get('release_date', {}).get('date'),
                            'required_age': app_data.get('ratings', {}).get('dejus', {}).get('required_age', 'No especificado'),
                            'platforms': ', '.join(platforms_list),
                            'price_final_formatted': app_data.get('price_overview', {}).get('final_formatted', 'Gratis o sin precio'),
                            'supported_languages': app_data.get('supported_languages'),
                            'header_image': app_data.get('header_image'), 
                            'website': app_data.get('website')
                        }
                        await save_row_to_csv(game_details, file_lock)
            else:
                # Si falla, se reencola para que otro proxy lo intente, y se anota el fallo.
                if not limiter.record_failure():
                    await app_id_queue.put((app_id, proxy_url)) # Reencolamos la tarea
                else:
                    tqdm_asyncio.write(f"AVISO: Proxy {proxy_url.split('@')[-1]} descartado por fallos repetidos.")
        
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if not proxy_limiters[proxy_url].record_failure():
                await app_id_queue.put((app_id, proxy_url))
        except Exception as e:
            tqdm_asyncio.write(f"ERROR INESPERADO en worker {worker_id}: {e}")
        finally:
            pbar.update(1)
            app_id_queue.task_done()

async def main():
    # --- Carga y reanudación (sin cambios) ---
    processed_app_ids = set()
    if os.path.exists(OUTPUT_CSV_PATH):
        try:
            df_existente = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
            if 'appid' in df_existente.columns:
                processed_app_ids = set(df_existente['appid'].dropna().astype(int))
                print(f"Se reanudará el proceso. {len(processed_app_ids)} juegos ya han sido guardados.")
        except Exception:
            print("Archivo de salida vacío o corrupto. Empezando de cero.")

    proxies = load_proxies()
    if not proxies: sys.exit(1)

    app_ids = fetch_all_app_ids()
    if app_ids is None: sys.exit(1)
    
    app_ids_to_process = sorted(list(set(app_ids) - processed_app_ids))
    
    if not app_ids_to_process:
        print("¡Dataset completo!")
        return

    # --- Preparación del sistema de colas y limitadores ---
    proxy_limiters = {proxy: ProxyRateLimiter(proxy, REQUEST_INTERVAL_PER_PROXY) for proxy in proxies}
    
    app_id_queue = asyncio.Queue()
    from itertools import cycle
    proxy_cycler = cycle(proxies)
    for app_id in app_ids_to_process:
        await app_id_queue.put((app_id, next(proxy_cycler)))

    print(f"\n-> Tarea: {len(app_ids_to_process)} juegos restantes por procesar.")
    print(f"-> Estrategia: {CONCURRENT_REQUESTS} workers concurrentes con limitador de velocidad por proxy.")
    
    file_lock = asyncio.Lock()
    pbar = tqdm_asyncio(total=len(app_ids_to_process), desc="Procesando Tareas")

    connector = aiohttp.TCPConnector(ssl=False, limit_per_host=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Creamos los workers que consumirán de la cola
        workers = [asyncio.create_task(worker(i, app_id_queue, proxy_limiters, file_lock, session, pbar)) for i in range(CONCURRENT_REQUESTS)]
        
        # Esperamos a que la cola se vacíe
        await app_id_queue.join()

        # Cancelamos los workers que ya no tienen nada que hacer
        for w in workers:
            w.cancel()
    
    pbar.close()
    print(f"\n¡PROCESO DE SESIÓN COMPLETADO!")

if __name__ == "__main__":
    asyncio.run(main())