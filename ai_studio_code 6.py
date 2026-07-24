# =============================================================================
# MLB HOME RUN PREDICTION PLATFORM - PHASE 2: DATA PIPELINE (V2 ISOLATED)
# Granularity: Plate Appearance (PA) Level
# Output: v2_mlb_cache/pa_master_dataset.parquet
# =============================================================================

import os
import time
import logging
import hashlib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pyarrow as pa
import pyarrow.parquet as pq

# --- CONFIGURATION ---
SEASONS = [2022, 2023, 2024, 2025, 2026]
CACHE_DIR = "v2_mlb_cache"  # ISOLATED FROM V1
OUTPUT_FILE = os.path.join(CACHE_DIR, "pa_master_dataset.parquet")
MAX_WORKERS = 8
CURRENT_DATE = datetime.now().date()

os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- STADIUM & PHYSICS CONSTANTS ---
STADIUM_DATA = {
    109: {"team": "ARI", "lat": 33.445, "lon": -112.067, "elev": 331, "dome": True},
    144: {"team": "ATL", "lat": 33.891, "lon": -84.468,  "elev": 305, "dome": False},
    110: {"team": "BAL", "lat": 39.284, "lon": -76.622,  "elev": 13,  "dome": False},
    111: {"team": "BOS", "lat": 42.346, "lon": -71.097,  "elev": 6,   "dome": False},
    112: {"team": "CHC", "lat": 41.948, "lon": -87.656,  "elev": 181, "dome": False},
    145: {"team": "CWS", "lat": 41.830, "lon": -87.634,  "elev": 181, "dome": False},
    113: {"team": "CIN", "lat": 39.097, "lon": -84.507,  "elev": 148, "dome": False},
    114: {"team": "CLE", "lat": 41.496, "lon": -81.685,  "elev": 202, "dome": False},
    115: {"team": "COL", "lat": 39.756, "lon": -104.994, "elev": 1580,"dome": False},
    116: {"team": "DET", "lat": 42.339, "lon": -83.049,  "elev": 183, "dome": False},
    117: {"team": "HOU", "lat": 29.757, "lon": -95.356,  "elev": 12,  "dome": True},
    118: {"team": "KC",  "lat": 39.051, "lon": -94.480,  "elev": 268, "dome": False},
    108: {"team": "LAA", "lat": 33.800, "lon": -117.883, "elev": 48,  "dome": False},
    119: {"team": "LAD", "lat": 34.074, "lon": -118.240, "elev": 112, "dome": False},
    146: {"team": "MIA", "lat": 25.778, "lon": -80.220,  "elev": 3,   "dome": True},
    158: {"team": "MIL", "lat": 43.028, "lon": -87.971,  "elev": 181, "dome": True},
    142: {"team": "MIN", "lat": 44.982, "lon": -93.278,  "elev": 256, "dome": False},
    121: {"team": "NYM", "lat": 40.757, "lon": -73.846,  "elev": 4,   "dome": False},
    147: {"team": "NYY", "lat": 40.829, "lon": -73.926,  "elev": 9,   "dome": False},
    133: {"team": "ATH", "lat": 38.580, "lon": -121.514, "elev": 5,   "dome": False},
    143: {"team": "PHI", "lat": 39.906, "lon": -75.167,  "elev": 6,   "dome": False},
    134: {"team": "PIT", "lat": 40.447, "lon": -80.006,  "elev": 226, "dome": False},
    135: {"team": "SD",  "lat": 32.707, "lon": -117.157, "elev": 4,   "dome": False},
    137: {"team": "SF",  "lat": 37.779, "lon": -122.389, "elev": 5,   "dome": False},
    136: {"team": "SEA", "lat": 47.591, "lon": -122.333, "elev": 5,   "dome": True},
    138: {"team": "STL", "lat": 38.623, "lon": -90.193,  "elev": 140, "dome": False},
    139: {"team": "TB",  "lat": 27.768, "lon": -82.653,  "elev": 14,  "dome": True},
    140: {"team": "TEX", "lat": 32.751, "lon": -97.083,  "elev": 168, "dome": True},
    141: {"team": "TOR", "lat": 43.641, "lon": -79.390,  "elev": 78,  "dome": True},
    120: {"team": "WSH", "lat": 38.873, "lon": -77.007,  "elev": 9,   "dome": False},
}

PA_ENDING_EVENTS = {
    'single', 'double', 'triple', 'home_run', 'strikeout', 'walk', 'field_out',
    'grounded_into_double_play', 'force_out', 'hit_by_pitch', 'fielders_choice',
    'field_error', 'sac_fly', 'sac_bunt', 'intent_walk', 'strikeout_double_play',
    'double_play', 'catcher_interf', 'fielders_choice_out', 'sac_fly_double_play',
    'triple_play'
}

# --- SESSION MANAGER ---
def get_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

session = get_session()

# --- 1. CHADWICK BUREAU ID MAPPING ---
def fetch_chadwick_mapping():
    cache_path = os.path.join(CACHE_DIR, "chadwick_mapping.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    
    logger.info("Downloading Chadwick Bureau ID mapping...")
    url = "https://github.com/chadwickbureau/register/raw/master/data/people.csv"
    df = pd.read_csv(url, low_memory=False)
    
    cols = ['key_mlbam', 'key_retro', 'key_fangraphs', 'name_first', 'name_last']
    df = df[[c for c in cols if c in df.columns]].dropna(subset=['key_mlbam'])
    df['key_mlbam'] = df['key_mlbam'].astype(int)
    
    df.to_parquet(cache_path, index=False)
    return df

# --- 2. STATCAST PITCH-BY-PITCH INGESTION ---
def get_statcast_url(start_dt, end_dt):
    return (
        f"https://baseballsavant.mlb.com/statcast_search/csv?all=true"
        f"&hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfPull=&hfC="
        f"&hfSea=&hfSit=&player_type=batter&hfOuts=&opponent=&pitcher_throws=&batter_stands="
        f"&hfSA=&game_date_gt={start_dt}&game_date_lt={end_dt}&team=&position=&hfRO=&home_road="
        f"&hfFlag=&hfPull=&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name"
        f"&sort_col=pitches&player_event_sort=api_p_fl&sort_order=desc&min_pas=0&type=details&"
    )

def fetch_statcast_week(start_date, end_date):
    cache_key = hashlib.md5(f"statcast_{start_date}_{end_date}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.parquet")
    
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
        
    url = get_statcast_url(start_date, end_date)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        
        if "<html" in resp.text[:100].lower():
            logger.error(f"Received HTML instead of CSV for {start_date} to {end_date}")
            return pd.DataFrame()
            
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), low_memory=False)
        
        if not df.empty:
            df.to_parquet(cache_path, index=False)
            
        return df
    except Exception as e:
        logger.error(f"Failed to fetch Statcast {start_date} to {end_date}: {e}")
        return pd.DataFrame()

def build_statcast_history():
    intervals = []
    for year in SEASONS:
        start = datetime(year, 3, 20)
        end = datetime(year, 11, 5)
        if year == CURRENT_DATE.year:
            end = min(end, datetime.combine(CURRENT_DATE - timedelta(days=1), datetime.min.time()))
            
        curr = start
        while curr < end:
            next_dt = min(curr + timedelta(days=6), end)
            intervals.append((curr.strftime("%Y-%m-%d"), next_dt.strftime("%Y-%m-%d")))
            curr = next_dt + timedelta(days=1)
            
    logger.info(f"Generated {len(intervals)} weekly intervals for Statcast ingestion.")
    
    dfs = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_date = {executor.submit(fetch_statcast_week, s, e): (s, e) for s, e in intervals}
        for i, future in enumerate(as_completed(future_to_date)):
            s, e = future_to_date[future]
            try:
                df = future.result()
                if not df.empty:
                    dfs.append(df)
                if (i + 1) % 20 == 0:
                    logger.info(f"Processed {i + 1}/{len(intervals)} Statcast weeks...")
            except Exception as exc:
                logger.error(f"Interval {s} to {e} generated an exception: {exc}")
                
    if not dfs:
        raise ValueError("No Statcast data downloaded!")
        
    master_df = pd.concat(dfs, ignore_index=True)
    master_df = master_df.drop_duplicates(subset=['game_pk', 'at_bat_number', 'pitch_number'])
    return master_df

# --- 3. ENVIRONMENTAL PHYSICS ENGINE ---
def calculate_air_density(temp_c, pressure_hpa, dew_point_c, elevation_m):
    if pd.isna(temp_c) or pd.isna(pressure_hpa) or pd.isna(dew_point_c):
        return np.nan
    T_k = temp_c + 273.15
    p_v_hpa = 6.11 * (10 ** ((7.5 * dew_point_c) / (237.3 + dew_point_c)))
    p_v_pa = p_v_hpa * 100
    p_station_pa = pressure_hpa * 100
    p_d_pa = p_station_pa - p_v_pa
    R_d = 287.058
    R_v = 461.495
    rho = (p_d_pa / (R_d * T_k)) + (p_v_pa / (R_v * T_k))
    return rho

def fetch_historical_weather(lat, lon, date_str):
    cache_key = hashlib.md5(f"weather_{lat}_{lon}_{date_str}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.parquet")
    
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
        
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,"
        f"wind_speed_10m,wind_direction_10m&timezone=America%2FNew_York"
    )
    
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        if 'hourly' in data:
            df = pd.DataFrame(data['hourly'])
            df['date'] = date_str
            df.to_parquet(cache_path, index=False)
            return df
    except Exception as e:
        pass
    
    return pd.DataFrame()

def apply_environmental_physics(pa_df):
    logger.info("Applying Environmental Physics Engine...")
    unique_games = pa_df[['game_date', 'home_team']].drop_duplicates()
    
    weather_records = []
    for _, row in unique_games.iterrows():
        date_str = row['game_date']
        team_abbr = row['home_team']
        
        stadium = next((s for s in STADIUM_DATA.values() if s['team'] == team_abbr), None)
        if not stadium:
            continue
            
        if stadium['dome']:
            weather_records.append({
                'game_date': date_str, 'home_team': team_abbr,
                'air_density': 1.184, 'temperature_c': 22.2, 'wind_speed_mph': 0.0, 'wind_direction': 0.0
            })
            continue
            
        w_df = fetch_historical_weather(stadium['lat'], stadium['lon'], date_str)
        if not w_df.empty:
            temp_c = w_df['temperature_2m'].median()
            pressure = w_df['surface_pressure'].median()
            dew = w_df['dew_point_2m'].median()
            wind_kmh = w_df['wind_speed_10m'].median()
            wind_dir = w_df['wind_direction_10m'].median()
            
            rho = calculate_air_density(temp_c, pressure, dew, stadium['elev'])
            
            weather_records.append({
                'game_date': date_str, 'home_team': team_abbr,
                'air_density': rho, 'temperature_c': temp_c, 
                'wind_speed_mph': wind_kmh * 0.621371, 'wind_direction': wind_dir
            })
            
    weather_df = pd.DataFrame(weather_records)
    if not weather_df.empty:
        pa_df = pa_df.merge(weather_df, on=['game_date', 'home_team'], how='left')
        
    pa_df['air_density'] = pa_df['air_density'].fillna(1.184)
    pa_df['temperature_c'] = pa_df['temperature_c'].fillna(22.2)
    pa_df['wind_speed_mph'] = pa_df['wind_speed_mph'].fillna(0.0)
    pa_df['wind_direction'] = pa_df['wind_direction'].fillna(0.0)
    
    return pa_df

# --- 4. PA AGGREGATION ---
def aggregate_to_pa_level(raw_df):
    logger.info("Aggregating pitch-level data to PA-level...")
    raw_df = raw_df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
    pa_ending = raw_df[raw_df['events'].isin(PA_ENDING_EVENTS)].copy()
    
    pitch_counts = raw_df.groupby(['game_pk', 'at_bat_number']).size().reset_index(name='pa_pitch_count')
    raw_df['launch_speed'] = pd.to_numeric(raw_df['launch_speed'], errors='coerce')
    max_ev = raw_df.groupby(['game_pk', 'at_bat_number'])['launch_speed'].max().reset_index(name='pa_max_ev')
    
    pa_df = pa_ending.merge(pitch_counts, on=['game_pk', 'at_bat_number'], how='left')
    pa_df = pa_df.merge(max_ev, on=['game_pk', 'at_bat_number'], how='left')
    pa_df['is_hr'] = (pa_df['events'] == 'home_run').astype(int)
    
    cols_to_keep = [
        'game_date', 'game_pk', 'at_bat_number', 'inning', 'inning_topbot', 'outs_when_up',
        'batter', 'pitcher', 'stand', 'p_throws', 'events', 'is_hr',
        'pa_pitch_count', 'pa_max_ev', 'launch_angle', 'hit_distance_sc',
        'home_team', 'away_team', 'pitch_type', 'release_speed', 'release_spin_rate',
        'hc_x', 'hc_y', 'bb_type', 'vy0', 'vz0', 'bat_speed', 'swing_length', 'plate_x', 'plate_z'
    ]
    
    for c in cols_to_keep:
        if c not in pa_df.columns:
            pa_df[c] = np.nan
            
    pa_df = pa_df[cols_to_keep]
    return pa_df

def main():
    logger.info("Starting Phase 2: Data Pipeline (V2 ISOLATED)")
    chadwick_df = fetch_chadwick_mapping()
    raw_statcast = build_statcast_history()
    pa_df = aggregate_to_pa_level(raw_statcast)
    pa_df = apply_environmental_physics(pa_df)
    
    pa_df = pa_df.merge(
        chadwick_df[['key_mlbam', 'key_fangraphs', 'key_retro']], 
        left_on='batter', right_on='key_mlbam', how='left'
    ).rename(columns={'key_fangraphs': 'batter_fangraphs', 'key_retro': 'batter_retro'}).drop(columns=['key_mlbam'])
    
    pa_df = pa_df.merge(
        chadwick_df[['key_mlbam', 'key_fangraphs', 'key_retro']], 
        left_on='pitcher', right_on='key_mlbam', how='left'
    ).rename(columns={'key_fangraphs': 'pitcher_fangraphs', 'key_retro': 'pitcher_retro'}).drop(columns=['key_mlbam'])
    
    pa_df = pa_df.sort_values(['game_date', 'game_pk', 'at_bat_number']).reset_index(drop=True)
    pa_df = pa_df.drop_duplicates(subset=['game_pk', 'at_bat_number'])
    
    pa_df.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"Phase 2 Complete. Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()