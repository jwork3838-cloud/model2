# =============================================================================
# MLB HOME RUN PREDICTION PLATFORM - PHASE 5: BvP MODELING (V2 ISOLATED)
# Output: Updates v2_mlb_cache/features_master.parquet
# =============================================================================

import os
import logging
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
CACHE_DIR = "v2_mlb_cache" # ISOLATED FROM V1
PA_FILE = os.path.join(CACHE_DIR, "pa_master_dataset.parquet")
FEATURES_FILE = os.path.join(CACHE_DIR, "features_master.parquet")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PRIORS = {
    'hr_rate': 0.032, 'ev': 89.0, 'la': 12.0, 'hard_hit': 0.38,
    'barrel': 0.07, 'xHR': 0.035, 'velo': 93.0, 'plate_x': 0.0, 'plate_z': 2.5
}
STANDARD_PRIOR_WEIGHT = 25 

def calculate_bvp_features(pa_df):
    df = pa_df.copy()
    for col in ['plate_x', 'plate_z']:
        if col not in df.columns:
            df[col] = PRIORS[col]
            
    df['is_hard_hit'] = (df['pa_max_ev'] >= 95).astype(int)
    df['is_barrel'] = ((df['pa_max_ev'] >= 98) & (df['launch_angle'] >= 26) & (df['launch_angle'] <= 30)).astype(int)
    df['is_xHR'] = ((df['pa_max_ev'] >= 100) & (df['launch_angle'] >= 20) & (df['launch_angle'] <= 35)).astype(int)
    
    df['pitch_group'] = df['pitch_type'].map({
        'FF': 'FB', 'SI': 'FB', 'FC': 'FB',
        'SL': 'BR', 'CU': 'BR', 'ST': 'BR', 'SV': 'BR', 'KC': 'BR', 'CS': 'BR',
        'CH': 'OS', 'FS': 'OS', 'KN': 'OS', 'SC': 'OS', 'FO': 'OS'
    }).fillna('FB')
    
    df['hr_off_fb'] = ((df['is_hr'] == 1) & (df['pitch_group'] == 'FB')).astype(int)
    df['hr_off_br'] = ((df['is_hr'] == 1) & (df['pitch_group'] == 'BR')).astype(int)
    df['hr_off_os'] = ((df['is_hr'] == 1) & (df['pitch_group'] == 'OS')).astype(int)
    df['has_contact'] = df['pa_max_ev'].notna().astype(int)
    
    fill_cols = ['pa_max_ev', 'launch_angle', 'release_speed', 'plate_x', 'plate_z']
    for col in fill_cols:
        df[col] = df[col].fillna(0)

    df = df.sort_values(['batter', 'pitcher', 'game_date', 'game_pk', 'at_bat_number']).reset_index(drop=True)
    bvp_group = df.groupby(['batter', 'pitcher'])
    df['bvp_pa'] = bvp_group.cumcount()
    
    def get_shifted_cumsum(col):
        res = bvp_group[col].cumsum().shift(1)
        res.loc[df['bvp_pa'] == 0] = 0
        return res

    df['bvp_hr_sum'] = get_shifted_cumsum('is_hr')
    df['bvp_contact_count'] = get_shifted_cumsum('has_contact')
    df['bvp_ev_sum'] = get_shifted_cumsum('pa_max_ev')
    df['bvp_la_sum'] = get_shifted_cumsum('launch_angle')
    df['bvp_hard_hit_sum'] = get_shifted_cumsum('is_hard_hit')
    df['bvp_barrel_sum'] = get_shifted_cumsum('is_barrel')
    df['bvp_xHR_sum'] = get_shifted_cumsum('is_xHR')
    df['bvp_velo_sum'] = get_shifted_cumsum('release_speed')
    df['bvp_px_sum'] = get_shifted_cumsum('plate_x')
    df['bvp_pz_sum'] = get_shifted_cumsum('plate_z')
    df['bvp_fb_hr'] = get_shifted_cumsum('hr_off_fb')
    df['bvp_br_hr'] = get_shifted_cumsum('hr_off_br')
    df['bvp_os_hr'] = get_shifted_cumsum('hr_off_os')

    dynamic_hr_weight = np.maximum(5, 20 - (4 * df['bvp_hr_sum']))
    df['bvp_shrunk_hr_rate'] = (df['bvp_hr_sum'] + (PRIORS['hr_rate'] * dynamic_hr_weight)) / (df['bvp_pa'] + dynamic_hr_weight)
    
    W = STANDARD_PRIOR_WEIGHT
    df['bvp_shrunk_ev'] = (df['bvp_ev_sum'] + (PRIORS['ev'] * W)) / (df['bvp_contact_count'] + W)
    df['bvp_shrunk_la'] = (df['bvp_la_sum'] + (PRIORS['la'] * W)) / (df['bvp_contact_count'] + W)
    df['bvp_shrunk_hard_hit'] = (df['bvp_hard_hit_sum'] + (PRIORS['hard_hit'] * W)) / (df['bvp_contact_count'] + W)
    df['bvp_shrunk_barrel'] = (df['bvp_barrel_sum'] + (PRIORS['barrel'] * W)) / (df['bvp_contact_count'] + W)
    df['bvp_shrunk_xHR'] = (df['bvp_xHR_sum'] + (PRIORS['xHR'] * W)) / (df['bvp_contact_count'] + W)
    df['bvp_shrunk_velo'] = (df['bvp_velo_sum'] + (PRIORS['velo'] * W)) / (df['bvp_pa'] + W)
    df['bvp_shrunk_px'] = (df['bvp_px_sum'] + (PRIORS['plate_x'] * W)) / (df['bvp_pa'] + W)
    df['bvp_shrunk_pz'] = (df['bvp_pz_sum'] + (PRIORS['plate_z'] * W)) / (df['bvp_pa'] + W)

    bvp_features = df[[
        'game_pk', 'at_bat_number', 'batter', 'pitcher',
        'bvp_pa', 'bvp_shrunk_hr_rate', 'bvp_shrunk_ev', 'bvp_shrunk_la',
        'bvp_shrunk_hard_hit', 'bvp_shrunk_barrel', 'bvp_shrunk_xHR',
        'bvp_shrunk_velo', 'bvp_shrunk_px', 'bvp_shrunk_pz',
        'bvp_fb_hr', 'bvp_br_hr', 'bvp_os_hr'
    ]]
    return bvp_features

def main():
    logger.info("Starting Phase 5: BvP Modeling (V2 ISOLATED)")
    pa_df = pd.read_parquet(PA_FILE)
    features_df = pd.read_parquet(FEATURES_FILE)
    
    bvp_df = calculate_bvp_features(pa_df)
    
    existing_bvp_cols = [c for c in bvp_df.columns if c not in ['game_pk', 'at_bat_number', 'batter', 'pitcher']]
    features_df = features_df.drop(columns=[c for c in existing_bvp_cols if c in features_df.columns])
    
    features_df = features_df.merge(bvp_df, on=['game_pk', 'at_bat_number', 'batter', 'pitcher'], how='left')
    features_df.to_parquet(FEATURES_FILE, index=False)
    logger.info(f"Phase 5 Complete. Output saved to: {FEATURES_FILE}")

if __name__ == "__main__":
    main()