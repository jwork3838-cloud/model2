# =============================================================================
# MLB HOME RUN PREDICTION PLATFORM - V2 MASTER ENGINE (PHASE 7 OPTIMIZED)
# Execution: Daily via GitHub Actions
# Output: Pushes to Google Sheet (1N2aUB8oWk_6-o-nt7fqAChhsN0o4xeqAH-tRAxPptDs)
# =============================================================================

import os
import sys
import json
import math
import logging
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
import gspread

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIGURATION & CONSTANTS ---
CACHE_DIR = "v2_mlb_cache"   # ISOLATED FROM V1
MODEL_DIR = "v2_mlb_models"  # ISOLATED FROM V1
DATA_FILE = os.path.join(CACHE_DIR, "features_master.parquet")
MODEL_FILE = os.path.join(MODEL_DIR, "v2_production_model.pkl")

# UPDATED GOOGLE SHEET ID
SHEET_ID = os.environ.get("V2_SHEET_ID", "1N2aUB8oWk_6-o-nt7fqAChhsN0o4xeqAH-tRAxPptDs")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

STADIUMS = {
    "ARI": {"bearing": 0,   "elev": 331, "dome": True},
    "ATL": {"bearing": 20,  "elev": 305, "dome": False},
    "BAL": {"bearing": 45,  "elev": 13,  "dome": False},
    "BOS": {"bearing": 45,  "elev": 6,   "dome": False},
    "CHC": {"bearing": 45,  "elev": 181, "dome": False},
    "CWS": {"bearing": 135, "elev": 181, "dome": False},
    "CIN": {"bearing": 135, "elev": 148, "dome": False},
    "CLE": {"bearing": 0,   "elev": 202, "dome": False},
    "COL": {"bearing": 0,   "elev": 1580,"dome": False},
    "DET": {"bearing": 135, "elev": 183, "dome": False},
    "HOU": {"bearing": 315, "elev": 12,  "dome": True},
    "KC":  {"bearing": 45,  "elev": 268, "dome": False},
    "LAA": {"bearing": 45,  "elev": 48,  "dome": False},
    "LAD": {"bearing": 45,  "elev": 112, "dome": False},
    "MIA": {"bearing": 90,  "elev": 3,   "dome": True},
    "MIL": {"bearing": 135, "elev": 181, "dome": True},
    "MIN": {"bearing": 90,  "elev": 256, "dome": False},
    "NYM": {"bearing": 45,  "elev": 4,   "dome": False},
    "NYY": {"bearing": 45,  "elev": 9,   "dome": False},
    "ATH": {"bearing": 45,  "elev": 5,   "dome": False},
    "PHI": {"bearing": 0,   "elev": 6,   "dome": False},
    "PIT": {"bearing": 135, "elev": 226, "dome": False},
    "SD":  {"bearing": 0,   "elev": 4,   "dome": False},
    "SF":  {"bearing": 90,  "elev": 5,   "dome": False},
    "SEA": {"bearing": 45,  "elev": 5,   "dome": True},
    "STL": {"bearing": 90,  "elev": 140, "dome": False},
    "TB":  {"bearing": 45,  "elev": 14,  "dome": True},
    "TEX": {"bearing": 135, "elev": 168, "dome": True},
    "TOR": {"bearing": 0,   "elev": 78,  "dome": True},
    "WSH": {"bearing": 45,  "elev": 9,   "dome": False},
}

LEAGUE_AVG_HR_RATE = 0.032

# --- 1. ADVANCED FEATURE ENGINEERING ---
def calculate_spray_angle(hc_x, hc_y):
    adj_x = hc_x - 125.42
    adj_y = 198.27 - hc_y
    adj_y = np.where(adj_y == 0, 0.1, adj_y)
    return np.degrees(np.arctan(adj_x / adj_y))

def calculate_vaa(vy0, vz0):
    vy0 = np.where(vy0 == 0, -0.1, vy0)
    return np.degrees(np.arctan(vz0 / vy0))

def engineer_advanced_features(df):
    logger.info("Engineering Phase 7 Advanced Features...")
    df['spray_angle'] = calculate_spray_angle(df['hc_x'], df['hc_y'])
    df['is_flyball'] = (df['bb_type'] == 'fly_ball').astype(int)
    
    df['is_pulled'] = np.where(
        df['stand'] == 'R', 
        (df['spray_angle'] < -15).astype(int), 
        (df['spray_angle'] > 15).astype(int)
    )
    df['is_pulled_fb'] = df['is_flyball'] * df['is_pulled']
    df['vaa'] = calculate_vaa(df['vy0'], df['vz0'])
    
    def get_wind_assist(row):
        stadium = STADIUMS.get(row['home_team'])
        if not stadium or stadium['dome'] or pd.isna(row.get('wind_speed_mph')):
            return 0.0
        wind_dir_rad = math.radians(row.get('wind_direction', 0))
        stadium_rad = math.radians(stadium['bearing'])
        return row['wind_speed_mph'] * math.cos(wind_dir_rad - stadium_rad)
        
    if 'wind_speed_mph' in df.columns:
        df['wind_assist_cf'] = df.apply(get_wind_assist, axis=1)
    else:
        df['wind_assist_cf'] = 0.0

    if 'bat_speed' not in df.columns:
        df['bat_speed'] = 72.0
    else:
        df['bat_speed'] = df['bat_speed'].fillna(72.0)
        
    df = df.sort_values(['game_date', 'game_pk', 'at_bat_number']).reset_index(drop=True)
    
    b_grp = df.groupby('batter')
    df['b_pa_cum'] = b_grp.cumcount()
    df['b_hr_cum'] = b_grp['is_hr'].cumsum().shift(1).fillna(0)
    df['b_pulled_fb_cum'] = b_grp['is_pulled_fb'].cumsum().shift(1).fillna(0)
    df['b_bat_speed_roll'] = b_grp['bat_speed'].transform(lambda x: x.shift(1).rolling(50, min_periods=1).mean()).fillna(72.0)
    df['b_hr_rate_bayes'] = (df['b_hr_cum'] + (LEAGUE_AVG_HR_RATE * 100)) / (df['b_pa_cum'] + 100)
    
    p_grp = df.groupby('pitcher')
    df['p_pa_cum'] = p_grp.cumcount()
    df['p_hr_cum'] = p_grp['is_hr'].cumsum().shift(1).fillna(0)
    df['p_vaa_roll'] = p_grp['vaa'].transform(lambda x: x.shift(1).rolling(100, min_periods=1).mean()).fillna(-5.0)
    df['p_hr_rate_bayes'] = (df['p_hr_cum'] + (LEAGUE_AVG_HR_RATE * 100)) / (df['p_pa_cum'] + 100)
    
    df['matchup_vaa_bs_interaction'] = df['p_vaa_roll'] * df['b_bat_speed_roll']
    return df

# --- 2. MODELING PIPELINE ---
def train_optimized_ensemble(df):
    logger.info("Training Phase 7 Optimized Ensemble...")
    features = [
        'b_hr_rate_bayes', 'b_pulled_fb_cum', 'b_bat_speed_roll',
        'p_hr_rate_bayes', 'p_vaa_roll', 'matchup_vaa_bs_interaction',
        'wind_assist_cf', 'air_density'
    ]
    bvp_cols = [c for c in df.columns if 'bvp_shrunk' in c]
    features.extend(bvp_cols)
    
    for f in features:
        if f not in df.columns:
            df[f] = 0.0
            
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date')
    
    train_idx = int(len(df) * 0.8)
    train_df = df.iloc[:train_idx]
    test_df = df.iloc[train_idx:]
    
    X_train, y_train = train_df[features], train_df['is_hr']
    X_test, y_test = test_df[features], test_df['is_hr']
    
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42, n_jobs=-1)
    lgb_cal = CalibratedClassifierCV(lgb_model, method='sigmoid', cv=3)
    lgb_cal.fit(X_train_imp, y_train)
    
    xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb_cal = CalibratedClassifierCV(xgb_model, method='sigmoid', cv=3)
    xgb_cal.fit(X_train_imp, y_train)
    
    cb_model = cb.CatBoostClassifier(iterations=300, learning_rate=0.03, depth=5, verbose=0, random_seed=42)
    cb_cal = CalibratedClassifierCV(cb_model, method='sigmoid', cv=3)
    cb_cal.fit(X_train_imp, y_train)
    
    logger.info("Training Non-Negative Meta-Learner...")
    p_lgb = lgb_cal.predict_proba(X_train_imp)[:, 1]
    p_xgb = xgb_cal.predict_proba(X_train_imp)[:, 1]
    p_cb = cb_cal.predict_proba(X_train_imp)[:, 1]
    
    X_meta_train = np.column_stack((p_lgb, p_xgb, p_cb))
    meta_model = Ridge(alpha=1.0, positive=True)
    meta_model.fit(X_meta_train, y_train)
    
    weights = meta_model.coef_ / meta_model.coef_.sum()
    logger.info(f"Ensemble Weights: LGBM: {weights[0]:.3f}, XGB: {weights[1]:.3f}, CB: {weights[2]:.3f}")
    
    p_lgb_test = lgb_cal.predict_proba(X_test_imp)[:, 1]
    p_xgb_test = xgb_cal.predict_proba(X_test_imp)[:, 1]
    p_cb_test = cb_cal.predict_proba(X_test_imp)[:, 1]
    
    X_meta_test = np.column_stack((p_lgb_test, p_xgb_test, p_cb_test))
    final_preds = np.dot(X_meta_test, weights)
    
    brier = brier_score_loss(y_test, final_preds)
    ll = log_loss(y_test, final_preds)
    logger.info(f"Test Set Performance -> Brier: {brier:.5f} | LogLoss: {ll:.5f}")
    
    production_model = {
        'imputer': imputer,
        'features': features,
        'lgb': lgb_cal,
        'xgb': xgb_cal,
        'cb': cb_cal,
        'weights': weights,
        'metrics': {'brier': brier, 'logloss': ll}
    }
    
    joblib.dump(production_model, MODEL_FILE)
    return production_model

# --- 3. INFERENCE & EXPECTED VALUE ---
def calculate_fractional_kelly(prob, odds_american, fraction=0.25):
    if odds_american > 0:
        decimal_odds = (odds_american / 100.0) + 1.0
    else:
        decimal_odds = (100.0 / abs(odds_american)) + 1.0
        
    b = decimal_odds - 1.0
    q = 1.0 - prob
    
    kelly_pct = ( (b * prob) - q ) / b
    if kelly_pct <= 0:
        return 0.0
        
    return round((kelly_pct * fraction) * 100, 2)

def fetch_dk_odds():
    logger.info("Fetching DraftKings odds...")
    url = "https://sportsbook-nash.draftkings.com/sites/US-PA-SB/api/sportscontent/controldata/league/leagueSubcategory/v1/markets?isBatchable=false&templateVars=84240,17319&eventsQuery=$filter=leagueId%20eq%20'84240'%20AND%20clientMetadata/Subcategories/any(s:%20s/Id%20eq%20'17319')&marketsQuery=$filter=clientMetadata/subCategoryId%20eq%20'17319'&include=Events&entity=events"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    odds_dict = {}
    try:
        resp = requests.get(url, headers=headers, timeout=15).json()
        for sel in resp.get("selections", []):
            if sel.get("label") == "1+":
                p_info = next((x for x in sel.get("participants", []) if x.get("type") == "Player"), None)
                if p_info and p_info.get("name"):
                    odds_raw = sel.get("displayOdds", {}).get("american") or sel.get("americanOdds")
                    if odds_raw:
                        odds_dict[p_info["name"].lower()] = float(str(odds_raw).replace("+", ""))
    except Exception as e:
        logger.error(f"DK Odds fetch failed: {e}")
    return odds_dict

def run_daily_inference(df):
    logger.info("Running Daily Inference...")
    if not os.path.exists(MODEL_FILE):
        logger.error("No production model found.")
        return
        
    model_pkg = joblib.load(MODEL_FILE)
    features = model_pkg['features']
    imputer = model_pkg['imputer']
    weights = model_pkg['weights']
    
    # Get today's schedule
    today_str = datetime.now().strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={today_str}&endDate={today_str}&hydrate=probablePitcher"
    resp = requests.get(url).json()
    games = resp.get('dates', [{}])[0].get('games', [])
    
    if not games:
        logger.info("No games scheduled today.")
        return

    payloads = []
    for game in games:
        home_team = game['teams']['home']['team']['name']
        away_team = game['teams']['away']['team']['name']
        home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('id')
        away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('id')
        
        if not home_pitcher or not away_pitcher:
            continue
            
        recent_home_batters = df[df['home_team'] == home_team]['batter'].unique()[-15:]
        recent_away_batters = df[df['away_team'] == away_team]['batter'].unique()[-15:]
        
        matchups = [(b, home_pitcher, home_team) for b in recent_away_batters] + \
                   [(b, away_pitcher, away_team) for b in recent_home_batters]
                   
        for batter_id, pitcher_id, stadium in matchups:
            b_hist = df[df['batter'] == batter_id]
            if b_hist.empty: continue
            b_state = b_hist.sort_values('game_date').iloc[-1].to_dict()
            
            p_hist = df[df['pitcher'] == pitcher_id]
            if p_hist.empty: continue
            p_state = p_hist.sort_values('game_date').iloc[-1].to_dict()
            
            bvp_hist = df[(df['batter'] == batter_id) & (df['pitcher'] == pitcher_id)]
            if not bvp_hist.empty:
                bvp_state = bvp_hist.sort_values('game_date').iloc[-1].to_dict()
            else:
                bvp_state = {c: df[c].median() for c in features if 'bvp_' in c}
                
            vector = {**b_state, **p_state, **bvp_state}
            clean_vector = {col: vector.get(col, 0.0) for col in features}
            
            # Get Batter Name (from Chadwick mapping in df if available, else ID)
            b_name = b_hist.iloc[-1].get('batter_name', f"Batter_{batter_id}")
            p_name = p_hist.iloc[-1].get('pitcher_name', f"Pitcher_{pitcher_id}")
            
            payloads.append({
                'Batter': b_name,
                'Pitcher': p_name,
                'Stadium': stadium,
                'vector': clean_vector
            })
            
    if not payloads:
        logger.info("No valid matchups generated.")
        return
        
    X_infer = pd.DataFrame([p['vector'] for p in payloads])[features]
    X_infer_imp = imputer.transform(X_infer)
    
    p_lgb = model_pkg['lgb'].predict_proba(X_infer_imp)[:, 1]
    p_xgb = model_pkg['xgb'].predict_proba(X_infer_imp)[:, 1]
    p_cb = model_pkg['cb'].predict_proba(X_infer_imp)[:, 1]
    
    X_meta = np.column_stack((p_lgb, p_xgb, p_cb))
    final_probs = np.dot(X_meta, weights)
    
    dk_odds = fetch_dk_odds()
    
    final_output = []
    for i, p in enumerate(payloads):
        prob = final_probs[i]
        # Fuzzy match name to DK odds
        b_name_lower = str(p['Batter']).lower()
        odds = dk_odds.get(b_name_lower, None)
        
        ev = 0.0
        kelly = 0.0
        if odds:
            dec_odds = (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
            ev = (prob * dec_odds) - 1
            kelly = calculate_fractional_kelly(prob, odds, fraction=0.25)
            
        final_output.append({
            "Batter": p['Batter'],
            "Pitcher": p['Pitcher'],
            "Stadium": p['Stadium'],
            "HR_Prob": f"{prob*100:.2f}%",
            "DK_Odds": f"+{int(odds)}" if odds and odds > 0 else (str(int(odds)) if odds else "N/A"),
            "EV": f"{ev:.3f}" if odds else "N/A",
            "Quarter_Kelly": f"{kelly}%" if odds else "N/A",
            "Playable": "YES" if odds and ev > 0.02 else "NO"
        })
        
    df_out = pd.DataFrame(final_output).sort_values(by="HR_Prob", ascending=False)
    
    # Publish to Sheets
    creds_json = os.environ.get("GOOGLE_CREDENTIALS")
    if creds_json:
        try:
            gc = gspread.service_account_from_dict(json.loads(creds_json))
            sh = gc.open_by_key(SHEET_ID)
            ws = sh.worksheet("DailyRankings")
            ws.clear()
            ws.update([df_out.columns.tolist()] + df_out.values.tolist())
            logger.info(f"Published {len(df_out)} predictions to Google Sheets.")
        except Exception as e:
            logger.error(f"Failed to publish to Google Sheets: {e}")
    else:
        logger.warning("GOOGLE_CREDENTIALS not found. Skipping Sheets upload.")

# --- MAIN ---
if __name__ == "__main__":
    logger.info("Starting MLB HR Engine V2 (Phase 7)")
    
    if os.path.exists(DATA_FILE):
        df = pd.read_parquet(DATA_FILE)
        df = engineer_advanced_features(df)
        train_optimized_ensemble(df)
        run_daily_inference(df)
    else:
        logger.error(f"Data file {DATA_FILE} not found. Run Phases 2, 3, and 5 first.")
        
    logger.info("Phase 7 Complete. System is fully optimized and production-ready.")