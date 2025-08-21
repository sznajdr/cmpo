import streamlit as st
import json
import pandas as pd
import numpy as np
import requests
from collections import defaultdict, Counter
from datetime import datetime
import warnings
import io
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Team Analysis",
    page_icon="⚽",
    layout="wide"
)

def preprocess_csv(df):
    """
    Preprocessing function to clean and transform the uploaded CSV data
    """
    try:
        print("Starting CSV preprocessing...")
        
        # Store original column names before any changes
        original_cols = df.columns.tolist()
        # st.write(f"Original columns: {original_cols}") # Removed for cleaner Streamlit output

        # 1. Drop the 'Unnamed: 0' column if it exists and is an index artifact
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            # st.write("Dropped 'Unnamed: 0' column.") # Removed for cleaner Streamlit output

        # Store original column names after unnamed drop
        original_cols_after_unnamed_drop = df.columns.tolist()

        # 2. Combine 'current_club' and 'club' into a new 'club' column
        if 'current_club' in df.columns and 'club' in df.columns:
            df['club_combined'] = df['current_club'].fillna(df['club'])
            df = df.drop(columns=['current_club', 'club'])
            df = df.rename(columns={'club_combined': 'club'})
            # st.write("Combined 'current_club' and 'club' into 'club'.") # Removed for cleaner Streamlit output
        elif 'current_club' in df.columns:
            df = df.rename(columns={'current_club': 'club'})
            # st.write("Renamed 'current_club' to 'club'.") # Removed for cleaner Streamlit output

        # 3. Combine 'player' and 'player_name' into a new 'player_name' column
        if 'player' in df.columns and 'player_name' in df.columns:
            df['player_name_combined'] = df['player_name'].fillna(df['player'])
            df = df.drop(columns=['player', 'player_name'])
            df = df.rename(columns={'player_name_combined': 'player_name'})
            # st.write("Combined 'player' and 'player_name' into 'player_name'.") # Removed for cleaner Streamlit output
        elif 'player' in df.columns:
            df = df.rename(columns={'player': 'player_name'})
            # st.write("Renamed 'player' to 'player_name'.") # Removed for cleaner Streamlit output

        # 4. Handle 'comp_name' based on 'league_name'
        if 'comp_name' in df.columns:
            df = df.rename(columns={'comp_name': 'secondary_comp_name'})
            # st.write("Renamed 'comp_name' to 'secondary_comp_name'.") # Removed for cleaner Streamlit output

        # 5. Column reordering
        front_columns = [
            'league_name', 'data_type', 'club', 'player_name', 'position', 'age',
            'comp_url', 'player_url', 'injury', 'player_market_value',
            'country', 'nationality', 'second_nationality', 'league_id'
        ]

        current_columns = df.columns.tolist()

        # Identify columns that are not in the 'front_columns' list
        end_columns = []
        for col in original_cols_after_unnamed_drop:
            mapped_col = col
            if col == 'current_club' or col == 'club':
                mapped_col = 'club'
            elif col == 'player' or col == 'player_name':
                mapped_col = 'player_name'
            elif col == 'comp_name':
                mapped_col = 'secondary_comp_name'

            if mapped_col not in front_columns and mapped_col in current_columns and mapped_col not in end_columns:
                end_columns.append(mapped_col)

        # Filter out columns from front_columns that might not exist
        front_columns_filtered = [col for col in front_columns if col in df.columns]

        # Combine the lists to get the final desired order
        final_column_order = front_columns_filtered + end_columns

        # Reindex the DataFrame to apply the new column order
        df = df.reindex(columns=final_column_order)
        # st.write("Columns reordered successfully.") # Removed for cleaner Streamlit output

        # st.write("CSV preprocessing completed successfully!") # Removed for cleaner Streamlit output
        return df, True, "Preprocessing completed successfully!"

    except Exception as e:
        error_msg = f"Error during preprocessing: {str(e)}"
        # st.error(error_msg) # Removed for cleaner Streamlit output
        return df, False, error_msg

class EnhancedTeamTacticalPredictor:
    def __init__(self):
        self.data = None
        self.team_analysis = {}
        self.team_names = []
        self.position_map = {
            1: 'GK', 11: 'GK',
            30: 'SW', 31: 'SW', 32: 'RB', 33: 'RCB', 34: 'RCB', 35: 'CB', 36: 'LCB', 37: 'LCB',
            38: 'LB', 39: 'RWB', 40: 'LWB', 41: 'WB', 42: 'RWB', 48: 'LWB',
            60: 'DM', 61: 'DM', 63: 'CDM', 64: 'RDM', 65: 'DM', 66: 'LDM', 89: 'LDM', 90: 'RDM',
            55: 'ZM', 67: 'CM', 68: 'RCM', 69: 'LCM', 72: 'RZM', 73: 'RCM', 74: 'CM', 75: 'CM', 76: 'CM', 77: 'LCM',
            53: 'RM', 57: 'LOV', 71: 'ROV', 78: 'LM', 79: 'LV', 83: 'RM', 87: 'LM', 88: 'LM',
            82: 'AM', 84: 'AM', 85: 'CAM', 86: 'AM', 91: 'AM',
            92: 'SS', 93: 'CF', 100: 'FW', 101: 'ST', 102: 'ST', 103: 'RW', 104: 'RS', 105: 'ST',
            106: 'LS', 107: 'LW', 115: 'ST'
        }

    def load_optimized_data(self, json_url):
        """Load pre-processed optimized data from a JSON file"""
        try:
            st.write(f"🔍 Trying to load data from: {json_url}")
            
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(json_url, timeout=60, headers=headers)
            
            if response.status_code == 200:
                st.write(f"✅ File found, size: {len(response.content)} bytes")
                
                # Debug: Show first few bytes
                # first_bytes_str = response.content[:50].decode('utf-8', errors='ignore')
                # st.write(f"🔍 First 50 bytes: {first_bytes_str}...")
                
                raw_data = json.loads(response.content)
                st.write(f"✅ Successfully loaded JSON from {json_url}")
                
                # The provided optimized_football_data.json has a 'matches' key at the top level
                if isinstance(raw_data, dict) and 'matches' in raw_data:
                    self.data = raw_data['matches']
                    st.write(f"✅ Extracted {len(self.data)} matches from 'matches' key.")
                    
                    # Also load team names from the 'teams' key if available in the optimized format
                    if 'teams' in raw_data and raw_data['teams']:
                        self.team_names = sorted([team_info['name'] for team_info in raw_data['teams'].values()])
                        st.write(f"🏟️ Loaded {len(self.team_names)} team names from 'teams' key.")
                    else:
                        # Fallback to extracting team names from matches if 'teams' key is not present or empty
                        teams = set()
                        for match in self.data:
                            home_team = match.get('home_team')
                            away_team = match.get('away_team')
                            if home_team:
                                teams.add(home_team)
                            if away_team:
                                teams.add(away_team)
                        self.team_names = sorted(list(teams))
                        st.write(f"🏟️ Extracted {len(self.team_names)} team names from matches.")
                elif isinstance(raw_data, list): # Fallback if the JSON is just a list of matches
                    self.data = raw_data
                    st.write(f"✅ Loaded {len(self.data)} matches (JSON is a list).")
                    teams = set()
                    for match in self.data:
                        home_team = match.get('home_team')
                        away_team = match.get('away_team')
                        if home_team:
                            teams.add(home_team)
                        if away_team:
                            teams.add(away_team)
                    self.team_names = sorted(list(teams))
                    st.write(f"🏟️ Extracted {len(self.team_names)} team names from matches.")
                else:
                    st.error("❌ JSON data structure not recognized (expected 'matches' key or a list).")
                    return False

                if self.team_names:
                    # st.write(f"📋 Sample teams: {self.team_names[:5]}") # Removed for cleaner Streamlit output
                    return True
                else:
                    st.error("❌ No team names could be extracted from the data.")
                    return False
            else:
                st.error(f"❌ JSON file not found or accessible (status: {response.status_code})")
                return False

        except requests.exceptions.Timeout:
            st.error(f"❌ Request to {json_url} timed out.")
            return False
        except json.JSONDecodeError as jde:
            st.error(f"❌ JSON decoding error from {json_url}: {str(jde)}")
            st.write("Please ensure the JSON file is valid.")
            return False
        except Exception as e:
            st.error(f"❌ Error loading data from {json_url}: {str(e)}")
            return False

    def _safe_get(self, obj, path, default=None):
        """Safely get nested dictionary values"""
        keys = path.split('.')
        current = obj
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and key.isdigit():
                idx = int(key)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return default
            else:
                return default
        return current if current is not None else default

    def _parse_numeric_string(self, value):
        """Parse numeric string values"""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                if ' ' in value and '(' in value and ')' in value:
                    value_to_parse = value.split(' ')[0]
                else:
                    value_to_parse = value.replace('%', '')
                return float(value_to_parse)
            except ValueError:
                return 0.0
        return 0.0

    def analyze_team_tactical_profile(self, team_name):
        """Create comprehensive tactical profile for a specific team"""
        if not self.data:
            return None

        team_data = {
            'team_name': team_name,
            'matches': [],
            'formations': {},
            'player_pool': {},
            'performance_by_formation': {},
            'substitution_data': []
        }

        # Extract all matches for this team
        for match in self.data:
            match_info = self._extract_team_match_info(match, team_name)
            if match_info:
                team_data['matches'].append(match_info)

        if not team_data['matches']:
            return None

        # Sort matches by date
        team_data['matches'].sort(key=lambda x: x.get('date', ''))

        # Analyze formations
        self._analyze_team_formations(team_data)

        # Analyze player pool and rotation patterns
        self._analyze_player_rotations(team_data)

        # Analyze performance by formation
        self._analyze_formation_performance(team_data)

        # Analyze substitution patterns
        self._analyze_substitution_patterns(team_data)

        return team_data

    def _extract_team_match_info(self, match, team_name):
        """Extract match information for specific team"""
        # Optimized format directly provides these keys
        home_team = match.get('home_team')
        away_team = match.get('away_team')

        if team_name not in [home_team, away_team]:
            return None

        is_home = team_name == home_team
        opponent = away_team if is_home else home_team

        team_score = match.get('home_score') if is_home else match.get('away_score')
        opponent_score = match.get('away_score') if is_home else match.get('home_score')

        formation = match.get('home_formation') if is_home else match.get('away_formation')
        
        starters = match.get('home_lineup', []) if is_home else match.get('away_lineup', [])
        subs = match.get('home_subs', []) if is_home else match.get('away_subs', [])
        
        substitutions_data = match.get('substitutions', {})
        team_substitutions = substitutions_data.get('home', []) if is_home else substitutions_data.get('away', [])

        # --- MODIFICATION START ---
        # Extract and generalize stats directly here
        processed_stats = {}
        raw_match_stats = match.get('stats', {})
        for key, value in raw_match_stats.items():
            if is_home and key.startswith('home_'):
                processed_stats[key.replace('home_', '')] = value
            elif not is_home and key.startswith('away_'):
                processed_stats[key.replace('away_', '')] = value
        
        # Ensure common keys are present, even if 0, for consistent aggregation later
        # Example: xG might not be in the top-level stats, but is in player stats. 
        # For now, this just ensures the top-level team stat exists for consistency.
        # If actual xG is needed, it would require aggregating from player xG.
        if 'expected_goals_xg' not in processed_stats:
             processed_stats['expected_goals_xg'] = 0.0

        # --- MODIFICATION END ---

        return {
            'date': match.get('date'),
            'match_id': match.get('match_id'),
            'league': match.get('league'),
            'round': match.get('round'),
            'is_home': is_home,
            'opponent': opponent,
            'formation': formation,
            'starters': starters,
            'substitutes': subs,
            'substitutions': team_substitutions,
            'team_score': team_score,
            'opponent_score': opponent_score,
            'result': 'W' if team_score > opponent_score else 'D' if team_score == opponent_score else 'L',
            'stats': processed_stats # Pass the processed stats
        }

    def _extract_substitution_events(self, match, lineup_key, player_stats_dummy): # player_stats_dummy is unused now
        """
        This method is no longer strictly necessary with the optimized JSON,
        as 'substitutions' array directly provides sub-in events.
        It's kept for compatibility if needed for other data structures, but streamlined.
        """
        # In optimized JSON, match.substitutions directly contains player_id, player_name, minute
        # We can enrich it with position and stats if needed from lineup data
        
        # Example of how you might merge it if 'substitutions' only has minimal info:
        # For now, we assume the 'substitutions' list in the optimized JSON is sufficient
        return match.get('substitutions', {}).get(lineup_key, [])


    def _extract_player_info(self, players, player_stats_dummy): # player_stats_dummy is not used in optimized format
        """Extract detailed player information from optimized lineup structure"""
        player_info = []
        for player in players:
            player_data = {
                'id': player.get('id'),
                'name': player.get('name'),
                'position_id': player.get('position_id'),
                'position': player.get('position'),
                'shirt_number': player.get('shirt_number'),
                'age': player.get('age'),
                'stats': player.get('stats', {}), # Directly use stats
                'rating': player.get('rating', 0),
                'minutes': player.get('minutes', 0),
                'goals': player.get('goals', 0),
                'assists': player.get('assists', 0),
                'xG': player.get('xG', 0.0)
            }
            player_info.append(player_data)
        return player_info

    def _extract_single_player_info(self, player, player_stats_dummy): # player_stats_dummy not used
        """Extract single player information from optimized player dict"""
        return {
            'id': player.get('id'),
            'name': player.get('name'),
            'position_id': player.get('position_id'),
            'position': player.get('position'),
            'shirt_number': player.get('shirt_number'),
            'age': player.get('age'),
            'stats': player.get('stats', {}),
            'rating': player.get('rating', 0),
            'minutes': player.get('minutes', 0),
            'goals': player.get('goals', 0),
            'assists': player.get('assists', 0),
            'xG': player.get('xG', 0.0)
        }

    # Removed _extract_team_match_stats as its functionality is integrated into _extract_team_match_info

    def _analyze_team_formations(self, team_data):
        """Analyze formation usage patterns"""
        formation_usage = defaultdict(int)
        formation_performance = defaultdict(lambda: {'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0})

        for match in team_data['matches']:
            formation = match.get('formation')
            if formation:
                formation_usage[formation] += 1
                formation_performance[formation][match['result']] += 1
                formation_performance[formation]['GF'] += match['team_score']
                formation_performance[formation]['GA'] += match['opponent_score']

        total_matches = len(team_data['matches'])

        for formation, count in formation_usage.items():
            perf = formation_performance[formation]
            total = perf['W'] + perf['D'] + perf['L']

            team_data['formations'][formation] = {
                'usage_count': count,
                'usage_rate': (count / total_matches) * 100,
                'wins': perf['W'],
                'draws': perf['D'],
                'losses': perf['L'],
                'win_rate': (perf['W'] / total) * 100 if total > 0 else 0,
                'points_per_game': (perf['W'] * 3 + perf['D']) / total if total > 0 else 0,
                'goals_for_avg': perf['GF'] / total if total > 0 else 0,
                'goals_against_avg': perf['GA'] / total if total > 0 else 0
            }

    def _analyze_player_rotations(self, team_data):
        """Analyze enhanced player rotation patterns"""
        player_appearances = defaultdict(lambda: {
            'name': '',
            'starts': 0,
            'sub_appearances': 0,
            'formations_played': defaultdict(int),
            'positions_played': defaultdict(int),
            'recent_starts': [],
            'performance': [],
            'goals': 0,
            'assists': 0,
            'xG': 0.0,
            'total_minutes': 0,
            'sub_minutes': [],
            'total_rating': 0,
            'rating_count': 0
        })

        total_matches = len(team_data['matches'])

        for i, match in enumerate(team_data['matches']):
            # Track starters
            for player in match.get('starters', []):
                player_id = player['id']
                data = player_appearances[player_id]

                data['name'] = player['name']
                data['starts'] += 1
                data['formations_played'][match.get('formation', '')] += 1
                data['positions_played'][player.get('position', '')] += 1
                data['recent_starts'].append(i)
                data['goals'] += player.get('goals', 0)
                data['assists'] += player.get('assists', 0)
                data['xG'] += player.get('xG', 0.0)
                data['total_minutes'] += player.get('minutes', 0)

                if player.get('rating', 0) > 0:
                    data['total_rating'] += player['rating']
                    data['rating_count'] += 1
                    data['performance'].append(player['rating'])

            # Track substitutes that actually played
            # For optimized JSON, players in 'home_subs'/'away_subs' lists
            # only appear if they actually subbed in.
            for player in match.get('substitutes', []):
                player_id = player['id']
                data = player_appearances[player_id]

                data['name'] = player['name']
                # Check if this player was explicitly a substitute in the 'substitutions' list for this match
                is_subbed_in = any(s['player_id'] == player_id for s in match.get('substitutions', []))
                if is_subbed_in: # Only count if they actually subbed in
                    data['sub_appearances'] += 1
                    data['goals'] += player.get('goals', 0)
                    data['assists'] += player.get('assists', 0)
                    data['xG'] += player.get('xG', 0.0)
                    data['total_minutes'] += player.get('minutes', 0)

                    if player.get('rating', 0) > 0:
                        data['total_rating'] += player['rating']
                        data['rating_count'] += 1
                        data['performance'].append(player['rating'])

        # Calculate comprehensive player metrics
        for player_id, data in player_appearances.items():
            total_apps = data['starts'] + data['sub_appearances']
            start_rate = data['starts'] / total_matches if total_matches > 0 else 0

            # Determine player role
            if total_apps == 0: # Handle players who didn't appear at all but are in lineup data
                role = "⚪ Non-playing"
            elif start_rate > 0.8:
                role = "🔵 Key Player"
            elif start_rate > 0.5:
                role = "🟡 Regular Starter"
            elif start_rate > 0.2:
                role = "🟠 Squad Rotation"
            else:
                role = "⚪ Fringe Player"

            # Calculate recent form
            recent_starts = data['recent_starts'][-5:] if data['recent_starts'] else []
            recent_frequency = len(recent_starts) / min(5, total_matches) if total_matches > 0 else 0

            team_data['player_pool'][player_id] = {
                'name': data['name'],
                'total_appearances': total_apps,
                'starts': data['starts'],
                'sub_appearances': data['sub_appearances'],
                'start_rate': start_rate * 100,
                'role': role,
                'primary_formation': max(data['formations_played'], key=data['formations_played'].get) if data['formations_played'] else '',
                'primary_position': max(data['positions_played'], key=data['positions_played'].get) if data['positions_played'] else '',
                'recent_frequency': recent_frequency * 100,
                'avg_rating': data['total_rating'] / data['rating_count'] if data['rating_count'] > 0 else 0,
                'recent_form_avg': np.mean(data['performance'][-5:]) if data['performance'] else 0,
                'recent_starts': recent_starts,
                'goals': data['goals'],
                'assists': data['assists'],
                'xG': data['xG'],
                'total_minutes': data['total_minutes'],
                'minutes_per_game': data['total_minutes'] / total_apps if total_apps > 0 else 0,
                'avg_sub_minute': round(np.mean(data['sub_minutes']), 1) if data['sub_minutes'] else 0,
                'sub_minute_range': f"{min(data['sub_minutes'])}-{max(data['sub_minutes'])}'" if data['sub_minutes'] else ''
            }

    def _analyze_substitution_patterns(self, team_data):
        """Analyze detailed substitution patterns"""
        substitution_analysis = defaultdict(lambda: {
            'total_sub_apps': 0,
            'goals_as_sub': 0,
            'assists_as_sub': 0,
            'xG_as_sub': 0.0,
            'avg_sub_minute': 0,
            'sub_minutes': [],
            'results_when_subbed': {'W': 0, 'D': 0, 'L': 0},
            'avg_rating_as_sub': 0,
            'total_rating_as_sub': 0,
            'rating_count_as_sub': 0
        })

        for match in team_data['matches']:
            # The optimized JSON has a 'substitutions' block directly under the match.
            # This contains 'player_id', 'player_name', 'minute'.
            # We need to find the full player data from the 'substitutes' list
            # to get their stats like goals, assists, xG, rating.
            
            team_substitutions_in_match = match.get('substitutions', [])
            
            # Map player_id to full player data for easy lookup
            sub_players_data = {player['id']: player for player in match.get('substitutes', [])}

            for sub_event in team_substitutions_in_match:
                player_id = sub_event['player_id']
                
                # Retrieve full player stats if available
                player_full_data = sub_players_data.get(player_id)

                if player_full_data: # Only process if player data is available
                    sa = substitution_analysis[player_id]

                    sa['total_sub_apps'] += 1
                    sa['results_when_subbed'][match['result']] += 1

                    # Track substitution timing
                    sub_minute = sub_event.get('minute', 'Unknown')
                    if sub_minute != 'Unknown':
                        try:
                            minute_num = int(str(sub_minute).replace("'", "").split("+")[0])
                            sa['sub_minutes'].append(minute_num)
                        except:
                            pass

                    # Track performance as substitute using full player data
                    sa['goals_as_sub'] += player_full_data.get('goals', 0)
                    sa['assists_as_sub'] += player_full_data.get('assists', 0)
                    sa['xG_as_sub'] += player_full_data.get('xG', 0.0)

                    rating = player_full_data.get('rating', 0)
                    if rating > 0:
                        sa['total_rating_as_sub'] += rating
                        sa['rating_count_as_sub'] += 1

        # Calculate averages
        for player_id, sa in substitution_analysis.items():
            if sa['sub_minutes']:
                sa['avg_sub_minute'] = round(np.mean(sa['sub_minutes']), 1)
            if sa['rating_count_as_sub'] > 0:
                sa['avg_rating_as_sub'] = round(sa['total_rating_as_sub'] / sa['rating_count_as_sub'], 2)
            
            # Add player name to substitution_analysis for easy lookup
            # This requires iterating player_pool or getting player name from initial subs list
            if player_id in team_data['player_pool']:
                sa['name'] = team_data['player_pool'][player_id]['name']
            else:
                # Fallback if player_id wasn't in main player_pool (e.g., only appeared as unused sub)
                # This could be handled by looking at the original match.substitutes list for name
                found_name = None
                for match in team_data['matches']:
                    for sub_player in match.get('substitutes', []):
                        if sub_player.get('id') == player_id: # Use .get to avoid KeyError
                            found_name = sub_player.get('name')
                            break
                    if found_name:
                        break
                sa['name'] = found_name if found_name else f"Player {player_id}"


        team_data['substitution_analysis'] = substitution_analysis


    def _analyze_formation_performance(self, team_data):
        """Analyze detailed performance by formation"""
        for formation, data in team_data['formations'].items():
            formation_matches = [m for m in team_data['matches'] if m.get('formation') == formation]

            if formation_matches:
                # Calculate advanced stats
                avg_stats = {}
                # These keys should directly match the processed_stats keys from _extract_team_match_info
                stat_keys = ['ball_possession', 'total_shots', 'shots_on_target',
                           'big_chances', 'accurate_passes', 'fouls_committed', 'corners', 'expected_goals_xg']

                for key in stat_keys:
                    # --- MODIFICATION START ---
                    # Now 'match['stats']' already contains the generic keys like 'ball_possession'
                    values = [m['stats'].get(key, 0) for m in formation_matches] # No need for 'if key in m.get('stats', {})'
                    # --- MODIFICATION END ---
                    avg_stats[key] = np.mean(values) if values else 0

                team_data['performance_by_formation'][formation] = {
                    'matches': len(formation_matches),
                    'avg_xG': avg_stats.get('expected_goals_xg', 0), 
                    'avg_possession': avg_stats.get('ball_possession', 0),
                    'avg_shots': avg_stats.get('total_shots', 0),
                    'avg_shots_on_target': avg_stats.get('shots_on_target', 0),
                    'avg_big_chances': avg_stats.get('big_chances', 0),
                    'avg_accurate_passes': avg_stats.get('accurate_passes', 0),
                    'avg_fouls': avg_stats.get('fouls_committed', 0),
                    'avg_corners': avg_stats.get('corners', 0),
                    'style_profile': self._determine_formation_style(data, avg_stats)
                }

    def _determine_formation_style(self, formation_data, avg_stats):
        """Determine playing style for formation"""
        possession = avg_stats.get('ball_possession', 0)
        goals_avg = formation_data.get('goals_for_avg', 0)

        if possession > 55:
            if goals_avg > 1.5:
                return "🎯 Possession Attack"
            else:
                return "🔄 Possession Control"
        elif avg_stats.get('total_shots', 0) > 13:
            return "🚀 Direct Attack"
        elif formation_data.get('goals_against_avg', 0) < 1.0:
            return "🛡️ Defensive Solid"
        else:
            return "⚖️ Balanced"

    def create_team_report(self, team_name):
        """Create comprehensive team tactical report"""
        team_data = self.analyze_team_tactical_profile(team_name)

        if not team_data:
            return f"❌ No data found for {team_name}"

        report = []
        report.append(f"🏆 {team_name.upper()} - COMPREHENSIVE TACTICAL ANALYSIS")
        report.append("=" * 80)

        # 1. SQUAD ROTATION ANALYSIS
        report.append(f"\n🔄 SQUAD ROTATION ANALYSIS")
        report.append("-" * 50)

        # Group players by role
        player_roles = defaultdict(list)
        for player_id, data in team_data['player_pool'].items():
            if data['name'] and data['total_appearances'] > 0: # Only include players who actually appeared
                player_roles[data['role']].append(data)

        # Ensure consistent order of roles for display
        role_order = ["🔵 Key Player", "🟡 Regular Starter", "🟠 Squad Rotation", "⚪ Fringe Player", "⚪ Non-playing"]
        
        for role in role_order:
            players = player_roles.get(role, [])
            if players:
                report.append(f"\n{role} ({len(players)} players):")
                # Sort players within each role for consistent output
                # Prioritize by starts, then total minutes, then avg_rating
                sorted_players = sorted(players, key=lambda x: (x['starts'], x['total_minutes'], x['avg_rating']), reverse=True)
                for player in sorted_players[:20]: # Limit to top 20 per role for brevity
                    # Format comprehensive player stats
                    stats_parts = []

                    # Goals and assists
                    if player['goals'] > 0 or player['assists'] > 0:
                        stats_parts.append(f"{player['goals']}G+{player['assists']}A")

                    # Rating
                    if player['avg_rating'] > 0:
                        stats_parts.append(f"{player['avg_rating']:.1f}★")

                    # Minutes per game
                    if player['minutes_per_game'] > 0:
                        stats_parts.append(f"{player['minutes_per_game']:.0f}min/game")

                    # Substitution info
                    if player['sub_appearances'] > 0:
                        sub_info = f"{player['sub_appearances']}Sub"
                        if player['avg_sub_minute'] > 0:
                            sub_info += f"@{player['avg_sub_minute']:.0f}'"
                        stats_parts.append(sub_info)

                    # Recent form
                    if player['recent_form_avg'] > 0:
                        stats_parts.append(f"Form:{player['recent_form_avg']:.1f}")

                    stats_display = " | " + " | ".join(stats_parts) if stats_parts else ""
                    
                    # Ensure position is displayed properly, handle empty string
                    position_display = f"({player['primary_position']})" if player['primary_position'] else ""

                    report.append(f"  • {player['name']} {position_display}: "
                                  f"{player['starts']}S+{player['sub_appearances']}Sub ({player['start_rate']:.0f}%){stats_display}")

        # 2. DETAILED FORMATION PERFORMANCE
        report.append(f"\n📊 DETAILED FORMATION PERFORMANCE")
        report.append("-" * 60)

        sorted_formations = sorted(
            team_data['formations'].items(),
            key=lambda x: x[1]['usage_count'],
            reverse=True
        )

        for formation, data in sorted_formations:
            perf_data = team_data['performance_by_formation'].get(formation, {})
            report.append(f"\n🏟️ {formation} Formation ({data['usage_count']} matches)")
            report.append(f"   📈 Record: {data['wins']}W-{data['draws']}D-{data['losses']}L ({data['win_rate']:.1f}% win rate)")
            report.append(f"   ⚽ Goals: {data['goals_for_avg']:.1f} for, {data['goals_against_avg']:.1f} against per game")
            report.append(f"   📊 Points per Game: {data['points_per_game']:.2f}")

            if perf_data:
                report.append(f"   📈 Advanced Stats:")
                report.append(f"      • xG: {perf_data.get('avg_xG', 0):.2f} per game") 
                report.append(f"      • Possession: {perf_data.get('avg_possession', 0):.1f}%")
                report.append(f"      • Shots: {perf_data.get('avg_shots', 0):.1f} per game")
                report.append(f"      • Shots on Target: {perf_data.get('avg_shots_on_target', 0):.1f} per game")
                report.append(f"      • Big Chances: {perf_data.get('avg_big_chances', 0):.1f} per game")
                report.append(f"      • Accurate Passes: {perf_data.get('avg_accurate_passes', 0):.0f} per game")
                report.append(f"      • Fouls: {perf_data.get('avg_fouls', 0):.1f} per game")
                report.append(f"      • Corners: {perf_data.get('avg_corners', 0):.1f} per game")
                report.append(f"      • Style: {perf_data.get('style_profile', 'Unknown')}")

        return "\n".join(report)


# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = EnhancedTeamTacticalPredictor()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'csv_preprocessing_done' not in st.session_state:
    st.session_state.csv_preprocessing_done = False

# Auto-load JSON data on startup
if not st.session_state.data_loaded:
    json_url = "https://raw.githubusercontent.com/sznajdr/cmpo/main/optimized_football_data.json"
    if st.session_state.analyzer.load_optimized_data(json_url):
        st.session_state.data_loaded = True

# Create tabs
tab1, tab2 = st.tabs(["Team Analysis", "Player Data"])

with tab1:
    # Main interface
    if st.session_state.data_loaded and st.session_state.analyzer.team_names:
        # Team selection
        selected_team = st.selectbox(
            "Select team:",
            options=st.session_state.analyzer.team_names,
            key="team_selector"
        )
        
        if selected_team:
            if st.button("Analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    report = st.session_state.analyzer.create_team_report(selected_team)
                
                st.code(report, language=None)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"{selected_team.replace(' ', '_')}_analysis.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    team_data = st.session_state.analyzer.analyze_team_tactical_profile(selected_team)
                    if team_data:
                        csv_data = []
                        for player_id, player in team_data['player_pool'].items():
                            if player['total_appearances'] > 0: # Only include players who appeared
                                csv_data.append({
                                    'Player': player['name'],
                                    'Position': player['primary_position'], 
                                    'Starts': player['starts'],
                                    'Sub Apps': player['sub_appearances'],
                                    'Start Rate %': round(player['start_rate'], 1),
                                    'Goals': player['goals'],
                                    'Assists': player['assists'],
                                    'Avg Rating': round(player['avg_rating'], 2),
                                    'Minutes/Game': round(player['minutes_per_game'], 0),
                                    'Role': player['role']
                                })
                        
                        csv_df = pd.DataFrame(csv_data)
                        csv_string = csv_df.to_csv(index=False)
                        
                        st.download_button(
                            label="Download CSV",
                            data=csv_string,
                            file_name=f"{selected_team.replace(' ', '_')}_data.csv",
                            mime="text/csv"
                        )
    else:
        st.info("No team analysis data loaded. Please ensure the data source is correct and accessible.")

with tab2:
    # CSV Upload Section
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            with st.spinner("Processing..."):
                processed_df, success, _ = preprocess_csv(df)
                st.session_state.csv_data = processed_df
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.csv_data = None
    
    # Auto-load default CSV if no file uploaded
    elif st.session_state.csv_data is None:
        try:
            csv_url = "https://raw.githubusercontent.com/sznajdr/cmpo/main/fdmbl.csv"
            df = pd.read_csv(csv_url)
            processed_df, _, _ = preprocess_csv(df)
            st.session_state.csv_data = processed_df
        except:
            # Fallback sample data (should ideally load the remote CSV)
            sample_data = [
                {
                    'league_name': 'Jupiler Pro League',
                    'data_type': 'injuries',
                    'club': 'Club Brugge KV',
                    'player_name': 'Gustaf Nilsson',
                    'position': 'Centre-Forward',
                    'age': 28,
                    'nationality': 'Sweden',
                    'player_market_value': 5000000,
                    'injury': 'Achilles tendon problems'
                }
            ]
            st.session_state.csv_data = pd.DataFrame(sample_data)
    
    # Display data and filters
    if st.session_state.csv_data is not None and not st.session_state.csv_data.empty:
        # Create filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # League filter
            try:
                leagues = ['All'] + sorted(st.session_state.csv_data['league_name'].dropna().unique().tolist())
            except:
                leagues = ['All']
            selected_league = st.selectbox("League:", leagues)
        
        with col2:
            # Data type filter
            try:
                data_types = ['All'] + sorted(st.session_state.csv_data['data_type'].dropna().unique().tolist())
            except:
                data_types = ['All']
            selected_data_type = st.selectbox("Type:", data_types)
        
        with col3:
            # Club filter
            try:
                clubs = ['All'] + sorted(st.session_state.csv_data['club'].dropna().unique().tolist())
            except:
                clubs = ['All']
            selected_club = st.selectbox("Club:", clubs)
        
        with col4:
            # Position filter - multi-select
            try:
                all_positions = sorted(st.session_state.csv_data['position'].dropna().unique().tolist())
                selected_positions = st.multiselect(
                    "Position(s):", 
                    all_positions,
                    placeholder="Select positions..."
                )
            except:
                selected_positions = []
        
        # Additional filters
        col5, col6, col7 = st.columns(3)
        
        with col5:
            # Age range
            try:
                if 'age' in st.session_state.csv_data.columns and st.session_state.csv_data['age'].notna().any():
                    min_age = int(st.session_state.csv_data['age'].min()) if not pd.isna(st.session_state.csv_data['age'].min()) else 16
                    max_age = int(st.session_state.csv_data['age'].max()) if not pd.isna(st.session_state.csv_data['age'].max()) else 40
                    age_range = st.slider("Age Range:", min_age, max_age, (min_age, max_age))
                else:
                    age_range = (16, 40)
            except:
                age_range = (16, 40)
        
        with col6:
            # Market value range (in millions)
            try:
                if 'player_market_value' in st.session_state.csv_data.columns and st.session_state.csv_data['player_market_value'].notna().any():
                    min_value = 0
                    # Max value needs to handle very large numbers correctly, possibly using ceil
                    max_raw_value = st.session_state.csv_data['player_market_value'].max()
                    if pd.notna(max_raw_value) and max_raw_value > 0:
                        max_value = int(np.ceil(max_raw_value / 1000000))
                        if max_value == 0 and max_raw_value > 0: # Handle cases like 0.5M
                            max_value = 1
                    else:
                        max_value = 100 # Default max if no valid data
                    
                    value_range = st.slider("Market Value (M€):", min_value, max_value, (min_value, max_value))
                else:
                    value_range = (0, 100)
            except Exception as e:
                # st.error(f"Error setting market value slider: {e}") # Removed for cleaner Streamlit output
                value_range = (0, 100)
        
        with col7:
            # Search by player name
            search_name = st.text_input("Search Player:", placeholder="Enter player name...")
        
        # Apply filters
        filtered_df = st.session_state.csv_data.copy()
        
        try:
            if selected_league != 'All' and 'league_name' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['league_name'] == selected_league]
            
            if selected_data_type != 'All' and 'data_type' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['data_type'] == selected_data_type]
            
            if selected_club != 'All' and 'club' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['club'] == selected_club]
            
            if selected_positions and 'position' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['position'].isin(selected_positions)]
            
            if 'age' in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['age'].between(age_range[0], age_range[1])) | 
                    (filtered_df['age'].isna())
                ]
            
            if 'player_market_value' in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['player_market_value'].between(value_range[0] * 1000000, value_range[1] * 1000000)) | 
                    (filtered_df['player_market_value'].isna())
                ]
            
            if search_name and 'player_name' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['player_name'].str.contains(search_name, case=False, na=False)
                ]
        except Exception as e:
            st.warning(f"Filter application error: {e}")
        
        # Format market value for display
        display_df = filtered_df.copy()
        try:
            if 'player_market_value' in display_df.columns:
                display_df['market_value_formatted'] = display_df['player_market_value'].apply(
                    lambda x: f"€{x/1000000:.1f}M" if pd.notna(x) and x > 0 else "-"
                )
                # Replace original 'player_market_value' with formatted one
                if 'Market Value' in display_df.columns:
                    display_df['Market Value'] = display_df['market_value_formatted']
                else:
                    display_df = display_df.rename(columns={'player_market_value': 'Market Value'})
                    display_df['Market Value'] = display_df['market_value_formatted']
                display_df = display_df.drop(columns=['market_value_formatted'])
        except:
            pass
        
        # Display results
        st.write(f"Showing {len(filtered_df)} of {len(st.session_state.csv_data)} players")
        
        if not filtered_df.empty:
            # Select columns to display
            # Ensure 'market_value_formatted' is used if created
            all_possible_columns_display = [
                'player_name', 'club', 'position', 'age', 
                'nationality', 'league_name', 'Market Value', # Use 'Market Value' for display
                'data_type', 'injury'
            ]
            
            available_columns_for_display = [col for col in all_possible_columns_display if col in display_df.columns]
            
            # If after filtering, there are still no common display columns, pick some default
            if not available_columns_for_display:
                available_columns_for_display = display_df.columns.tolist()[:6]
            
            # Rename columns for display before showing
            column_names_for_display = {
                'player_name': 'Player', 'club': 'Club', 'position': 'Position',
                'age': 'Age', 'nationality': 'Nationality', 'league_name': 'League',
                'data_type': 'Type', 'injury': 'Injury' # Market Value already handled if it exists
            }
            
            display_df_show = display_df[available_columns_for_display].copy()
            for old_name, new_name in column_names_for_display.items():
                if old_name in display_df_show.columns:
                    display_df_show = display_df_show.rename(columns={old_name: new_name})
            
            st.dataframe(display_df_show, use_container_width=True, hide_index=True)
            
            # Download button for the filtered data
            csv_download = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered CSV",
                data=csv_download,
                file_name="filtered_player_data.csv",
                mime="text/csv"
            )
        else:
            st.info("No players match the selected filters")
