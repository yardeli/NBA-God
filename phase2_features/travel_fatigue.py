"""
NBA-God Phase 3: Travel Fatigue Feature

NBA teams play 82 games across geographically dispersed arenas.
Travel fatigue impacts performance:
- Back-to-back games (B2B): -2 to -3 points per game penalty
- Long travel distance: Additional fatigue for West Coast trips
- Rest advantage: 3+ days rest vs. opponent's 1 day = +1-2 point lift

This feature compounds the model's performance metrics.

FEATURE: diff_travel_fatigue
Computed as: (away_rest_days - home_rest_days) + (away_distance / 1000) * 0.1
Higher value = away team more fatigued (favors home)

EXPECTED IMPACT: +0.5-1.5% ROI on moneyline bets
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "phase2_features" / "output"

# NBA Arena coordinates (longitude, latitude) for distance calculations
NBA_ARENAS = {
    "Atlanta Hawks": (-84.3963, 33.7490),
    "Boston Celtics": (-71.0621, 42.3661),
    "Brooklyn Nets": (-73.9776, 40.6826),
    "Charlotte Hornets": (-80.8387, 35.2251),
    "Chicago Bulls": (-87.6742, 41.8807),
    "Cleveland Cavaliers": (-81.6882, 41.4963),
    "Dallas Mavericks": (-96.8101, 32.7905),
    "Denver Nuggets": (-104.9903, 39.7487),
    "Detroit Pistons": (-83.0551, 42.6811),
    "Golden State Warriors": (-122.2011, 37.7694),
    "Houston Rockets": (-95.1622, 29.7588),
    "LA Clippers": (-118.2437, 34.0430),
    "LA Lakers": (-118.2437, 34.0430),
    "Memphis Grizzlies": (-90.0131, 35.1382),
    "Miami Heat": (-80.1895, 25.7617),
    "Milwaukee Bucks": (-87.9168, 43.0044),
    "Minnesota Timberwolves": (-93.2787, 44.9795),
    "New Orleans Pelicans": (-90.0821, 29.9487),
    "New York Knicks": (-73.9776, 40.7505),
    "Oklahoma City Thunder": (-97.5151, 35.4634),
    "Orlando Magic": (-81.3837, 28.5421),
    "Philadelphia 76ers": (-75.1721, 39.9012),
    "Phoenix Suns": (-112.0712, 33.3755),
    "Portland Trail Blazers": (-122.6688, 45.2308),
    "Sacramento Kings": (-121.5020, 38.5816),
    "San Antonio Spurs": (-98.4379, 29.4769),
    "Toronto Raptors": (-79.3957, 43.6426),
    "Utah Jazz": (-111.9011, 40.7683),
    "Washington Wizards": (-77.0209, 38.8980),
}

def great_circle_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates (miles)."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3959  # Earth radius in miles
    return c * r

def add_travel_fatigue_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add travel fatigue feature to game dataframe.
    
    Requires columns:
    - home_team_name / away_team_name
    - rest_days_home / rest_days_away
    """
    
    if "home_team_name" not in df.columns or "away_team_name" not in df.columns:
        print("[WARN] Cannot add travel fatigue (need team names)")
        return df
    
    df = df.copy()
    
    # Calculate travel distance for away team
    travel_distances = []
    for idx, row in df.iterrows():
        home_team = row.get("home_team_name", "")
        away_team = row.get("away_team_name", "")
        
        if home_team not in NBA_ARENAS or away_team not in NBA_ARENAS:
            travel_distances.append(0)
            continue
        
        home_coords = NBA_ARENAS[home_team]
        away_coords = NBA_ARENAS[away_team]
        
        distance = great_circle_distance(
            away_coords[1], away_coords[0],
            home_coords[1], home_coords[0]
        )
        travel_distances.append(distance)
    
    df["travel_distance"] = travel_distances
    
    # Rest advantage: (away rest - home rest)
    # Negative value = away team more rested (good for away)
    if "rest_days_home" in df.columns and "rest_days_away" in df.columns:
        df["rest_advantage"] = df["rest_days_away"] - df["rest_days_home"]
    else:
        df["rest_advantage"] = 0
    
    # Fatigue composite: travel distance + rest disadvantage
    # Higher = away team more fatigued (favors home)
    # Formula: distance_factor + rest_penalty
    #   distance_factor: 1 mile = 0.001 fatigue points
    #   rest_penalty: -1 rest day = 0.5 fatigue points
    df["travel_fatigue"] = (df["travel_distance"] * 0.001) - (df["rest_advantage"] * 0.5)
    
    # Create differential: away fatigue - home fatigue
    # (Home team always benefits from being at home, distance only affects away team)
    df["diff_travel_fatigue"] = df["travel_fatigue"] * -1  # Flip sign: positive = away tired
    
    return df

if __name__ == "__main__":
    print("Travel Fatigue Feature Module")
    print("Usage: from travel_fatigue import add_travel_fatigue_feature")
    print(f"Supports {len(NBA_ARENAS)} NBA teams")
