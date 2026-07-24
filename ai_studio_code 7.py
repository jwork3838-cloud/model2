# =============================================================================
# MLB HOME RUN PREDICTION PLATFORM - PHASE 3: FEATURE ENGINEERING (V2 ISOLATED)
# Output: v2_mlb_cache/features_master.parquet
# =============================================================================

import os
import logging
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
CACHE_DIR = "v2_mlb_cache" # ISOLATED FROM V1
INPUT_FILE = os.path.join(CACHE_DIR, "pa_master_dataset.parquet")
OUTPUT_FILE = os.path.join(CACHE_DIR, "features_master.parquet")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LEAGUE_AVG_HR_RATE = 0.032
PRIOR_PA = 100

PITCH_GROUPS = {
    'FF': 'FB', 'SI': 'FB', 'FC': 'FB',
    'SL': 'BR', 'CU': 'BR', 'ST': 'BR', 'SV': 'BR', 'KC': 'BR', 'CS': 'BR',
    'CH': 'OS', 'FS': 'OS', 'KN': 'OS', 'SC': 'OS', 'FO': 'OS'
}

def bayesian_shrinkage(successes, trials, prior_rate=LEAGUE_AVG_HR_RATE, prior_trials=PRIOR_PA):
    return (successes + (prior_rate * prior_trials)) / (trials + prior_trials)

def safe_divide(num, denom, default=0.0):
    return np.where(denom > 0, num / denom, default)

def engineer_pitch_types(df):
    df['pitch_group'] = df['pitch_type'].map(PITCH_GROUPS).fillna('FB')
    df['is_FB'] = (df['pitch_group'] == 'FB').astype(int)
    df['is_BR'] = (df['pitch_group'] == 'BR').astype(int)
    df['is_OS'] = (df['pitch_group'] == 'OS').astype(int)
    df['hr_FB'] = df['is_hr'] * df['is_FB']
    df['hr_BR'] = df['is_hr'] * df['is_BR']
    df['hr_OS'] = df['is_hr'] * df['is_OS']
    return df

def build_batter_features(df):
    b_game = df.groupby(['batter', 'game_date', 'game_pk']).agg(
        pa_count=('is_hr', 'count'),
        hr_count=('is_hr', 'sum'),
        fb_seen=('is_FB', 'sum'),
        br_seen=('is_BR', 'sum'),
        os_seen=('is_OS', 'sum'),
        fb_hr=('hr_FB', 'sum'),
        br_hr=('hr_BR', 'sum'),
        os_hr=('hr_OS', 'sum')
    ).reset_index().sort_values(['batter', 'game_date'])
    
    b_game_shifted = b_game.groupby('batter').shift(1)
    b_game_shifted['batter'] = b_game['batter']
    b_game_shifted['game_pk'] = b_game['game_pk']
    
    b_roll = b_game_shifted.groupby('batter').rolling(window=150, min_periods=1).sum().reset_index(level=0, drop=True)
    b_roll_50 = b_game_shifted.groupby('batter').rolling(window=50, min_periods=1).sum().reset_index(level=0, drop=True)
    
    features = pd.DataFrame({
        'batter': b_game_shifted['batter'],
        'game_pk': b_game_shifted['game_pk'],
        'b_hr_rate_150': bayesian_shrinkage(b_roll['hr_count'], b_roll['pa_count']),
        'b_fb_hr_rate': bayesian_shrinkage(b_roll['fb_hr'], b_roll['fb_seen']),
        'b_br_hr_rate': bayesian_shrinkage(b_roll['br_hr'], b_roll['br_seen']),
        'b_os_hr_rate': bayesian_shrinkage(b_roll['os_hr'], b_roll['os_seen']),
        'b_hr_rate_50': bayesian_shrinkage(b_roll_50['hr_count'], b_roll_50['pa_count']),
    })
    features['b_hr_trend'] = features['b_hr_rate_50'] - features['b_hr_rate_150']
    return df.merge(features, on=['batter', 'game_pk'], how='left')

def build_pitcher_features(df):
    df['game_date_dt'] = pd.to_datetime(df['game_date'])
    p_game = df.groupby(['pitcher', 'game_date_dt', 'game_pk']).agg(
        pa_count=('is_hr', 'count'),
        hr_count=('is_hr', 'sum'),
        pitch_count=('pa_pitch_count', 'sum'),
        fb_thrown=('is_FB', 'sum'),
        br_thrown=('is_BR', 'sum'),
        os_thrown=('is_OS', 'sum')
    ).reset_index().sort_values(['pitcher', 'game_date_dt'])
    
    p_game_shifted = p_game.groupby('pitcher').shift(1)
    p_game_shifted['pitcher'] = p_game['pitcher']
    p_game_shifted['game_pk'] = p_game['game_pk']
    p_game_shifted['game_date_dt'] = p_game['game_date_dt']
    
    p_roll = p_game_shifted.groupby('pitcher').rolling(window=100, min_periods=1).sum().reset_index(level=0, drop=True)
    
    p_game_shifted = p_game_shifted.set_index('game_date_dt')
    p_fatigue = p_game_shifted.groupby('pitcher')['pitch_count'].rolling('5D').sum().reset_index(level=0, drop=True)
    p_game_shifted = p_game_shifted.reset_index()
    
    p_game_shifted['last_game_date'] = p_game_shifted.groupby('pitcher')['game_date_dt'].shift(1)
    p_game_shifted['days_rest'] = (p_game_shifted['game_date_dt'] - p_game_shifted['last_game_date']).dt.days.fillna(99)
    p_game_shifted['days_rest'] = np.clip(p_game_shifted['days_rest'], 0, 10)
    
    total_pitches = p_roll['fb_thrown'] + p_roll['br_thrown'] + p_roll['os_thrown']
    
    features = pd.DataFrame({
        'pitcher': p_game_shifted['pitcher'],
        'game_pk': p_game_shifted['game_pk'],
        'p_hr_rate_100': bayesian_shrinkage(p_roll['hr_count'], p_roll['pa_count']),
        'p_fb_pct': safe_divide(p_roll['fb_thrown'], total_pitches, 0.55),
        'p_br_pct': safe_divide(p_roll['br_thrown'], total_pitches, 0.30),
        'p_os_pct': safe_divide(p_roll['os_thrown'], total_pitches, 0.15),
        'p_recent_pitches_5d': p_fatigue.values,
        'p_days_rest': p_game_shifted['days_rest']
    })
    return df.merge(features, on=['pitcher', 'game_pk'], how='left')

def build_matchup_interactions(df):
    df['p_fb_pct'] = df['p_fb_pct'].fillna(0.55)
    df['p_br_pct'] = df['p_br_pct'].fillna(0.30)
    df['p_os_pct'] = df['p_os_pct'].fillna(0.15)
    df['b_fb_hr_rate'] = df['b_fb_hr_rate'].fillna(LEAGUE_AVG_HR_RATE)
    df['b_br_hr_rate'] = df['b_br_hr_rate'].fillna(LEAGUE_AVG_HR_RATE)
    df['b_os_hr_rate'] = df['b_os_hr_rate'].fillna(LEAGUE_AVG_HR_RATE)
    
    df['matchup_exp_hr_rate'] = (
        (df['p_fb_pct'] * df['b_fb_hr_rate']) +
        (df['p_br_pct'] * df['b_br_hr_rate']) +
        (df['p_os_pct'] * df['b_os_hr_rate'])
    )
    return df

def main():
    logger.info("Starting Phase 3: Feature Engineering (V2 ISOLATED)")
    df = pd.read_parquet(INPUT_FILE)
    df = engineer_pitch_types(df)
    df = build_batter_features(df)
    df = build_pitcher_features(df)
    df = build_matchup_interactions(df)
    
    cols_to_drop = ['pitch_group', 'is_FB', 'is_BR', 'is_OS', 'hr_FB', 'hr_BR', 'hr_OS', 'game_date_dt']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    df.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"Phase 3 Complete. Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()