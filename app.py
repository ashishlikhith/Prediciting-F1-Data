"""
F1 Race Prediction & Analysis API
Flask backend with multiple ML models for predicting race outcomes
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# ─── Global Data ───────────────────────────────────────────────────────────────
df = None
le_gp = LabelEncoder()
le_cont = LabelEncoder()
le_team = LabelEncoder()
le_driver = LabelEncoder()
rf_reg = None
gb_reg = None
rf_clf = None
scaler = StandardScaler()

# ─── Helper: Convert time string to seconds ───────────────────────────────────
def time_to_seconds(t):
    try:
        t = str(t).strip()
        parts = t.split(':')
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0.0
            m, s = parts
        else:
            return np.nan
        return int(h * 3600 + m * 60 + s)
    except Exception:
        return np.nan

# ─── Load and Preprocess Data ──────────────────────────────────────────────────
def load_data():
    global df
    csv_path = os.path.join(os.path.dirname(__file__), 'f1_dataset.csv')
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    needed_cols = ['time', 'laps', 'year', 'team', 'grand_prix', 'continent', 'winner_name']
    df = df.dropna(subset=needed_cols).copy()
    df['laps'] = pd.to_numeric(df['laps'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['laps', 'year']).copy()
    df['time_seconds'] = df['time'].apply(time_to_seconds)
    df = df.dropna(subset=['time_seconds']).copy()
    df['time_seconds'] = df['time_seconds'].astype(int)

    # Encode categoricals
    df['gp_encoded'] = le_gp.fit_transform(df['grand_prix'].astype(str))
    df['continent_encoded'] = le_cont.fit_transform(df['continent'].astype(str))
    df['team_encoded'] = le_team.fit_transform(df['team'].astype(str))
    df['driver_encoded'] = le_driver.fit_transform(df['winner_name'].astype(str))

    print(f"Data loaded: {len(df)} records, {df['winner_name'].nunique()} drivers, "
          f"{df['grand_prix'].nunique()} tracks, {df['team'].nunique()} teams")

# ─── Train Models ──────────────────────────────────────────────────────────────
def train_models():
    global rf_reg, gb_reg, rf_clf

    features = ['laps', 'year', 'gp_encoded', 'continent_encoded', 'driver_encoded', 'team_encoded']
    X = df[features].copy()
    y_time = df['time_seconds'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y_time, test_size=0.2, random_state=42)

    # Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_train)
    rf_pred = rf_reg.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    print(f"RandomForest Regressor R²: {rf_r2:.4f}")

    # Gradient Boosting Regressor
    gb_reg = GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=5)
    gb_reg.fit(X_train, y_train)
    gb_pred = gb_reg.predict(X_test)
    gb_r2 = r2_score(y_test, gb_pred)
    print(f"GradientBoosting Regressor R²: {gb_r2:.4f}")

    # Random Forest Classifier for team prediction
    y_team = df['team_encoded'].copy()
    X_clf = df[['laps', 'year', 'time_seconds', 'gp_encoded', 'continent_encoded', 'driver_encoded']].copy()
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_team, test_size=0.2, random_state=42)
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_clf.fit(Xc_train, yc_train)
    clf_score = rf_clf.score(Xc_test, yc_test)
    print(f"RandomForest Classifier accuracy: {clf_score:.4f}")

# ─── Analysis Functions ────────────────────────────────────────────────────────

def get_driver_stats(driver_name):
    """Get comprehensive driver statistics."""
    ddf = df[df['winner_name'] == driver_name]
    if ddf.empty:
        return None

    total_wins = len(ddf)
    years_active = sorted(ddf['year'].unique().tolist())
    career_span = f"{int(min(years_active))}-{int(max(years_active))}" if years_active else "N/A"
    teams = ddf['team'].unique().tolist()
    tracks_won = ddf['grand_prix'].unique().tolist()

    # Wins per year
    wins_by_year = ddf.groupby('year').size().reset_index(name='wins')
    wins_by_year_list = [{"year": int(r['year']), "wins": int(r['wins'])} for _, r in wins_by_year.iterrows()]

    # Wins by continent
    wins_by_continent = ddf.groupby('continent').size().reset_index(name='wins')
    wins_by_continent_list = [{"continent": r['continent'], "wins": int(r['wins'])} for _, r in wins_by_continent.iterrows()]
    preferred_continent = wins_by_continent.loc[wins_by_continent['wins'].idxmax(), 'continent'] if not wins_by_continent.empty else "N/A"

    # Average race time
    avg_time = int(ddf['time_seconds'].mean()) if not ddf.empty else 0

    # Racing style analysis
    time_std = ddf['time_seconds'].std()
    avg_laps = ddf['laps'].mean()

    if total_wins >= 20:
        style = "Dominant"
    elif time_std < 1500:
        style = "Consistent"
    elif avg_laps > 65:
        style = "Endurance Specialist"
    else:
        style = "Strategic"

    # Strong tracks (won more than once)
    track_wins = ddf.groupby('grand_prix').size().reset_index(name='wins')
    strong_tracks = track_wins[track_wins['wins'] > 1].sort_values('wins', ascending=False)
    strong_tracks_list = [{"track": r['grand_prix'], "wins": int(r['wins'])} for _, r in strong_tracks.iterrows()]

    return {
        "name": driver_name,
        "total_wins": total_wins,
        "career_span": career_span,
        "years_active": [int(y) for y in years_active],
        "teams": teams,
        "tracks_won": tracks_won,
        "wins_by_year": wins_by_year_list,
        "wins_by_continent": wins_by_continent_list,
        "preferred_continent": preferred_continent,
        "avg_race_time_seconds": avg_time,
        "racing_style": style,
        "strong_tracks": strong_tracks_list[:10]
    }


def get_team_stats(team_name):
    """Get team performance statistics."""
    tdf = df[df['team'] == team_name]
    if tdf.empty:
        return None

    total_wins = len(tdf)
    years_active = sorted(tdf['year'].unique().tolist())
    career_span = f"{int(min(years_active))}-{int(max(years_active))}" if years_active else "N/A"
    drivers = tdf['winner_name'].unique().tolist()

    wins_by_year = tdf.groupby('year').size().reset_index(name='wins')
    wins_by_year_list = [{"year": int(r['year']), "wins": int(r['wins'])} for _, r in wins_by_year.iterrows()]

    # Best driver for team
    driver_wins = tdf.groupby('winner_name').size().reset_index(name='wins').sort_values('wins', ascending=False)
    top_drivers = [{"driver": r['winner_name'], "wins": int(r['wins'])} for _, r in driver_wins.head(5).iterrows()]

    return {
        "name": team_name,
        "total_wins": total_wins,
        "career_span": career_span,
        "years_active": [int(y) for y in years_active],
        "drivers": drivers,
        "wins_by_year": wins_by_year_list,
        "top_drivers": top_drivers
    }


def get_rivals(driver_name, top_n=5):
    """Find top rivals based on overlapping years and win counts."""
    ddf = df[df['winner_name'] == driver_name]
    if ddf.empty:
        return []

    driver_years = set(ddf['year'].unique())
    driver_wins = len(ddf)

    # Find drivers who raced in overlapping years
    rivals = []
    for other_driver in df['winner_name'].unique():
        if other_driver == driver_name:
            continue
        odf = df[df['winner_name'] == other_driver]
        other_years = set(odf['year'].unique())
        overlap = driver_years & other_years
        if len(overlap) >= 2:
            overlap_wins = len(odf[odf['year'].isin(overlap)])
            driver_overlap_wins = len(ddf[ddf['year'].isin(overlap)])
            rivals.append({
                "name": other_driver,
                "total_wins": len(odf),
                "overlap_years": len(overlap),
                "overlap_wins": overlap_wins,
                "driver_overlap_wins": driver_overlap_wins
            })

    # Sort by overlap wins descending
    rivals.sort(key=lambda x: x['overlap_wins'], reverse=True)
    return rivals[:top_n]


def predict_performance(driver_name, track_name, team_name, year):
    """Year-aware prediction combining historical form, track affinity, team strength, and ML models."""
    result = {}

    # ─── Year-scoped data slices ────────────────────────────────────────────
    up_to_year = df[df['year'] <= year]                 # all data up to & including selected year
    in_year    = df[df['year'] == year]                  # data for the exact year
    recent     = df[(df['year'] >= year - 2) & (df['year'] <= year)]  # 3-year window

    driver_all          = up_to_year[up_to_year['winner_name'] == driver_name]
    driver_at_track     = driver_all[driver_all['grand_prix'] == track_name]
    driver_recent       = recent[recent['winner_name'] == driver_name]
    driver_in_year      = in_year[in_year['winner_name'] == driver_name]
    track_up_to         = up_to_year[up_to_year['grand_prix'] == track_name]
    track_in_year       = in_year[in_year['grand_prix'] == track_name]
    team_up_to          = up_to_year[up_to_year['team'] == team_name]
    team_in_year        = in_year[in_year['team'] == team_name]
    driver_with_team    = driver_all[driver_all['team'] == team_name]

    # Also keep all-time for profile cards
    driver_alltime      = df[df['winner_name'] == driver_name]
    track_alltime       = df[df['grand_prix'] == track_name]

    # ─── Counts (year-scoped) ───────────────────────────────────────────────
    driver_track_wins   = len(driver_at_track)
    driver_total_wins   = len(driver_all)
    track_races         = len(track_up_to)
    team_wins           = len(team_up_to)
    team_wins_year      = len(team_in_year)
    driver_team_wins    = len(driver_with_team)
    driver_wins_year    = len(driver_in_year)
    driver_recent_wins  = len(driver_recent)
    races_in_year       = len(in_year)
    races_recent        = len(recent)

    # ─── 5-Factor Win Probability (all year-dependent) ──────────────────────

    # Factor 1: Driver momentum — how many wins in the 3-year window
    momentum_score = (driver_recent_wins / max(races_recent, 1)) * 100

    # Factor 2: Track affinity — driver wins at this track vs total track races so far
    track_affinity = (driver_track_wins / max(track_races, 1)) * 100

    # Factor 3: Team year-strength — how dominant is this team in the selected year
    team_year_strength = (team_wins_year / max(races_in_year, 1)) * 100

    # Factor 4: Driver-team synergy — driver wins with this team vs team total
    synergy = (driver_team_wins / max(team_wins, 1)) * 100 if team_wins > 0 else 0

    # Factor 5: Driver's year form — wins in the exact selected year
    year_form = (driver_wins_year / max(races_in_year, 1)) * 100

    # Weighted combination — year-specific factors get heavy weights
    win_probability = (
        momentum_score   * 0.20 +   # recent form matters
        track_affinity   * 0.20 +   # track-specific advantage
        team_year_strength * 0.25 + # car performance that year is crucial
        synergy          * 0.15 +   # driver-team chemistry
        year_form        * 0.20     # how the driver did THAT year
    )
    win_probability = round(min(max(win_probability, 0.5), 95.0), 1)

    # ─── Predicted Position (1-20, finer granularity) ───────────────────────
    if win_probability >= 55:
        predicted_position = 1
    elif win_probability >= 40:
        predicted_position = 2
    elif win_probability >= 28:
        predicted_position = 3
    elif win_probability >= 20:
        predicted_position = 4
    elif win_probability >= 14:
        predicted_position = 5
    elif win_probability >= 10:
        predicted_position = 6
    elif win_probability >= 7:
        predicted_position = 8
    elif win_probability >= 4:
        predicted_position = 10
    elif win_probability >= 2:
        predicted_position = 14
    else:
        predicted_position = 18

    # ─── Race time comparison — driver avg vs field avg that year ────────────
    driver_avg_time = int(driver_in_year['time_seconds'].mean()) if not driver_in_year.empty else None
    field_avg_time  = int(in_year['time_seconds'].mean()) if not in_year.empty else None
    time_advantage  = None
    if driver_avg_time and field_avg_time:
        time_advantage = field_avg_time - driver_avg_time  # positive = faster than field

    # ─── Podiums / wins per year breakdown (up to selected year) ────────────
    yearly_breakdown = []
    if not driver_all.empty:
        yb = driver_all.groupby('year').agg(
            wins=('winner_name', 'size'),
            avg_time=('time_seconds', 'mean'),
            races_team=('team', 'first')
        ).reset_index()
        for _, row in yb.iterrows():
            yearly_breakdown.append({
                "year": int(row['year']),
                "wins": int(row['wins']),
                "avg_time": int(row['avg_time']),
                "team": row['races_team']
            })

    # ─── Predict race time using ML models ──────────────────────────────────
    try:
        gp_enc = le_gp.transform([track_name])[0] if track_name in le_gp.classes_ else 0
        cont_val = track_alltime['continent'].mode().iloc[0] if not track_alltime.empty else 'Europe'
        cont_enc = le_cont.transform([cont_val])[0] if cont_val in le_cont.classes_ else 0
        team_enc = le_team.transform([team_name])[0] if team_name in le_team.classes_ else 0
        driver_enc = le_driver.transform([driver_name])[0] if driver_name in le_driver.classes_ else 0
        avg_laps = track_alltime['laps'].mean() if not track_alltime.empty else 60

        features = np.array([[avg_laps, year, gp_enc, cont_enc, driver_enc, team_enc]])
        rf_time = int(rf_reg.predict(features)[0])
        gb_time = int(gb_reg.predict(features)[0])

        def fmt(s):
            return f"{s//3600:01d}:{(s%3600)//60:02d}:{s%60:02d}"

        result['predicted_time_rf'] = fmt(rf_time)
        result['predicted_time_gb'] = fmt(gb_time)
        result['predicted_time_rf_seconds'] = rf_time
        result['predicted_time_gb_seconds'] = gb_time
    except Exception as e:
        result['predicted_time_rf'] = "N/A"
        result['predicted_time_gb'] = "N/A"
        result['prediction_error'] = str(e)

    # ─── Pack results ───────────────────────────────────────────────────────
    result['win_probability']       = win_probability
    result['predicted_position']    = predicted_position
    result['driver_track_wins']     = driver_track_wins
    result['driver_total_wins']     = driver_total_wins
    result['track_total_races']     = track_races
    result['team_total_wins']       = team_wins
    result['driver_team_wins']      = driver_team_wins
    result['track_win_rate']        = round(track_affinity, 1)
    result['team_synergy_rate']     = round(synergy, 1)

    # Year-specific extras
    result['selected_year']         = year
    result['driver_wins_year']      = driver_wins_year
    result['team_wins_year']        = team_wins_year
    result['races_in_year']         = races_in_year
    result['driver_recent_wins']    = driver_recent_wins
    result['momentum_score']        = round(momentum_score, 1)
    result['team_year_strength']    = round(team_year_strength, 1)
    result['year_form']             = round(year_form, 1)
    result['driver_avg_time_year']  = driver_avg_time
    result['field_avg_time_year']   = field_avg_time
    result['time_advantage']        = time_advantage
    result['yearly_breakdown']      = yearly_breakdown

    # Driver stats (all-time for profile)
    result['driver_stats'] = get_driver_stats(driver_name)
    result['team_stats']   = get_team_stats(team_name)
    result['rivals']       = get_rivals(driver_name)

    # Track history for this driver (up to selected year)
    if not driver_at_track.empty:
        result['track_history'] = [
            {"year": int(r['year']), "team": r['team'], "time": r['time'], "laps": int(r['laps'])}
            for _, r in driver_at_track.iterrows()
        ]
    else:
        result['track_history'] = []

    return result


# ─── API Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/drivers')
def api_drivers():
    drivers = sorted(df['winner_name'].unique().tolist())
    return jsonify(drivers)

@app.route('/api/tracks')
def api_tracks():
    tracks = sorted(df['grand_prix'].unique().tolist())
    return jsonify(tracks)

@app.route('/api/teams')
def api_teams():
    teams = sorted(df['team'].unique().tolist())
    return jsonify(teams)

@app.route('/api/years')
def api_years():
    years = sorted(df['year'].unique().astype(int).tolist())
    return jsonify(years)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    driver = data.get('driver', '')
    track = data.get('track', '')
    team = data.get('team', '')
    year = data.get('year', 2024)

    if not driver or not track or not team:
        return jsonify({"error": "driver, track, and team are required"}), 400

    try:
        year = int(year)
    except:
        year = 2024

    result = predict_performance(driver, track, team, year)
    return jsonify(result)

@app.route('/api/driver-stats/<name>')
def api_driver_stats(name):
    stats = get_driver_stats(name)
    if stats is None:
        return jsonify({"error": "Driver not found"}), 404
    return jsonify(stats)

@app.route('/api/team-stats/<name>')
def api_team_stats(name):
    stats = get_team_stats(name)
    if stats is None:
        return jsonify({"error": "Team not found"}), 404
    return jsonify(stats)


# ─── Startup ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading F1 data...")
    load_data()
    print("Training models...")
    train_models()
    print("Starting server on http://localhost:5000")
    app.run(debug=True, port=5000)
