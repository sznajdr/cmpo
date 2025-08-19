import streamlit as st
import json
import pandas as pd
import numpy as np
import requests
import pickle
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
        print(f"Original columns: {original_cols}")

        # 1. Drop the 'Unnamed: 0' column if it exists and is an index artifact
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            print("Dropped 'Unnamed: 0' column.")

        # Store original column names after unnamed drop
        original_cols_after_unnamed_drop = df.columns.tolist()

        # 2. Combine 'current_club' and 'club' into a new 'club' column
        if 'current_club' in df.columns and 'club' in df.columns:
            df['club_combined'] = df['current_club'].fillna(df['club'])
            df = df.drop(columns=['current_club', 'club'])
            df = df.rename(columns={'club_combined': 'club'})
            print("Combined 'current_club' and 'club' into 'club'.")
        elif 'current_club' in df.columns:
            df = df.rename(columns={'current_club': 'club'})
            print("Renamed 'current_club' to 'club'.")

        # 3. Combine 'player' and 'player_name' into a new 'player_name' column
        if 'player' in df.columns and 'player_name' in df.columns:
            df['player_name_combined'] = df['player_name'].fillna(df['player'])
            df = df.drop(columns=['player', 'player_name'])
            df = df.rename(columns={'player_name_combined': 'player_name'})
            print("Combined 'player' and 'player_name' into 'player_name'.")
        elif 'player' in df.columns:
            df = df.rename(columns={'player': 'player_name'})
            print("Renamed 'player' to 'player_name'.")

        # 4. Handle 'comp_name' based on 'league_name'
        if 'comp_name' in df.columns:
            df = df.rename(columns={'comp_name': 'secondary_comp_name'})
            print("Renamed 'comp_name' to 'secondary_comp_name'.")

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
        print("Columns reordered successfully.")

        print("CSV preprocessing completed successfully!")
        return df, True, "Preprocessing completed successfully!"

    except Exception as e:
        error_msg = f"Error during preprocessing: {str(e)}"
        print(error_msg)
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

    def load_optimized_data(self, base_url):
        """Load pre-processed optimized data from multiple PKL files"""
        try:
            # Remove trailing slash
            base_url = base_url.rstrip('/')
            
            combined_data = []
            file_count = 0
            
            # Try to load pkl1.pkl, pkl2.pkl, pkl3.pkl, etc.
            for i in range(1, 20):  # Try up to 20 files
                pkl_url = f"{base_url}/pkl{i}.pkl"
                
                try:
                    print(f"Trying to load: pkl{i}.pkl")
                    response = requests.get(pkl_url, timeout=60)
                    
                    if response.status_code == 200:
                        raw_data = pickle.loads(response.content)
                        
                        # Handle different data structures
                        if isinstance(raw_data, dict) and 'matches' in raw_data:
                            combined_data.extend(raw_data['matches'])
                        elif isinstance(raw_data, list):
                            combined_data.extend(raw_data)
                        
                        file_count += 1
                        print(f"Successfully loaded pkl{i}.pkl")
                    else:
                        # File doesn't exist, stop trying
                        break
                        
                except Exception as e:
                    # File doesn't exist or error, stop trying
                    break
            
            if combined_data:
                self.data = combined_data
                print(f"Successfully loaded {len(combined_data)} matches from {file_count} files")
                
                # Extract team names
                teams = set()
                for match in self.data:
                    home_team = self._safe_get(match, 'general.homeTeam.name')
                    away_team = self._safe_get(match, 'general.awayTeam.name')
                    if home_team:
                        teams.add(home_team)
                    if away_team:
                        teams.add(away_team)

                self.team_names = sorted(list(teams))
                return True
            else:
                print("No PKL files found")
                return False

        except Exception as e:
            print(f"Error loading data: {e}")
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
        home_team = self._safe_get(match, 'general.homeTeam.name')
        away_team = self._safe_get(match, 'general.awayTeam.name')

        if team_name not in [home_team, away_team]:
            return None

        is_home = team_name == home_team
        opponent = away_team if is_home else home_team

        # Get scores
        teams = self._safe_get(match, 'header.teams', [])
        if len(teams) < 2:
            return None

        team_score = teams[0]['score'] if is_home else teams[1]['score']
        opponent_score = teams[1]['score'] if is_home else teams[0]['score']

        # Get lineup
        lineup_key = 'homeTeam' if is_home else 'awayTeam'
        lineup_info = self._safe_get(match, f'content.lineup.{lineup_key}', {})

        formation = lineup_info.get('formation')
        starters = lineup_info.get('starters', [])
        subs = lineup_info.get('subs', [])

        # Get player stats
        player_stats = self._safe_get(match, 'content.playerStats', {})

        # Extract match stats
        stats = self._extract_team_match_stats(match, is_home)

        # Extract substitution events
        substitutions = self._extract_substitution_events(match, lineup_key, player_stats)

        return {
            'date': self._safe_get(match, 'general.matchTimeUTCDate', ''),
            'match_id': self._safe_get(match, 'general.matchId'),
            'league': self._safe_get(match, 'general.leagueName', ''),
            'round': self._safe_get(match, 'general.matchRound', ''),
            'is_home': is_home,
            'opponent': opponent,
            'formation': formation,
            'starters': self._extract_player_info(starters, player_stats),
            'substitutes': self._extract_player_info(subs, player_stats),
            'substitutions': substitutions,
            'team_score': team_score,
            'opponent_score': opponent_score,
            'result': 'W' if team_score > opponent_score else 'D' if team_score == opponent_score else 'L',
            'stats': stats
        }

    def _extract_substitution_events(self, match, lineup_key, player_stats):
        """Extract substitution timing and details"""
        lineup_info = self._safe_get(match, f'content.lineup.{lineup_key}', {})
        substitutes = lineup_info.get('subs', [])

        substitution_events = []

        for sub in substitutes:
            if isinstance(sub, dict):
                performance = sub.get('performance', {})
                sub_events = performance.get('substitutionEvents', [])

                for event in sub_events:
                    if event.get('type') == 'subIn':
                        sub_minute = event.get('time', 'Unknown')

                        # Extract player stats
                        player_info = self._extract_single_player_info(sub, player_stats)

                        substitution_events.append({
                            'player_id': str(sub.get('id', '')),
                            'player_name': sub.get('name', ''),
                            'position': player_info.get('position', ''),
                            'sub_minute': sub_minute,
                            'stats': player_info.get('stats', {}),
                            'rating': player_info.get('rating', 0),
                            'minutes_played': player_info.get('minutes', 0)
                        })
                        break

        return substitution_events

    def _extract_player_info(self, players, player_stats):
        """Extract detailed player information"""
        player_info = []
        for player in players:
            player_data = self._extract_single_player_info(player, player_stats)
            player_info.append(player_data)
        return player_info

    def _extract_single_player_info(self, player, player_stats):
        """Extract single player information with enhanced stats"""
        player_id = str(player.get('id', ''))
        position_id = player.get('positionId')
        position_label = self._safe_get(player, 'positionLabel.label', '')

        # Use position map if label not available
        if not position_label and position_id:
            position_label = self.position_map.get(position_id, 'Unknown')

        player_data = {
            'id': player_id,
            'name': player.get('name', ''),
            'position_id': position_id,
            'position': position_label,
            'shirt_number': player.get('shirtNumber'),
            'age': player.get('age'),
            'stats': {},
            'rating': 0,
            'minutes': 0,
            'goals': 0,
            'assists': 0,
            'xG': 0.0
        }

        # Add performance stats if available
        if player_id in player_stats:
            stats = player_stats[player_id].get('stats', [])
            for stat_group in stats:
                if stat_group.get('key') == 'top_stats':
                    top_stats = stat_group.get('stats', {})
                    player_data['stats'] = top_stats

                    # Extract key metrics
                    player_data['rating'] = self._safe_get(top_stats, 'FotMob rating.stat.value', 0.0)
                    player_data['minutes'] = self._safe_get(top_stats, 'Minutes played.stat.value', 0)
                    player_data['goals'] = self._safe_get(top_stats, 'Goals.stat.value', 0)
                    player_data['assists'] = self._safe_get(top_stats, 'Assists.stat.value', 0)
                    player_data['xG'] = self._safe_get(top_stats, 'Expected goals (xG).stat.value', 0.0)
                    break

        return player_data

    def _extract_team_match_stats(self, match, is_home):
        """Extract comprehensive team match statistics"""
        all_periods_stats = self._safe_get(match, 'content.stats.Periods.All.stats', [])
        team_stats = {}

        # Find top_stats group
        top_stats_group = next((g for g in all_periods_stats if g.get('key') == 'top_stats'), None)
        if top_stats_group:
            for stat_item in top_stats_group.get('stats', []):
                title = stat_item.get('title', '').lower()
                values = stat_item.get('stats', [])

                if len(values) >= 2:
                    team_value = self._parse_numeric_string(values[0]) if is_home else self._parse_numeric_string(values[1])
                    opponent_value = self._parse_numeric_string(values[1]) if is_home else self._parse_numeric_string(values[0])

                    # Store with clean key names
                    clean_title = title.replace('(', '').replace(')', '').replace(' ', '_')
                    team_stats[clean_title] = team_value
                    team_stats[f'opponent_{clean_title}'] = opponent_value

        return team_stats

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

            # Track substitutes
            for player in match.get('substitutes', []):
                player_id = player['id']
                data = player_appearances[player_id]

                data['name'] = player['name']
                data['sub_appearances'] += 1
                data['goals'] += player.get('goals', 0)
                data['assists'] += player.get('assists', 0)
                data['xG'] += player.get('xG', 0.0)
                data['total_minutes'] += player.get('minutes', 0)

                if player.get('rating', 0) > 0:
                    data['total_rating'] += player['rating']
                    data['rating_count'] += 1
                    data['performance'].append(player['rating'])

            # Track substitution events
            for sub_event in match.get('substitutions', []):
                player_id = sub_event['player_id']
                if player_id in player_appearances:
                    sub_minute = sub_event.get('sub_minute', 'Unknown')
                    if sub_minute != 'Unknown':
                        try:
                            minute_num = int(str(sub_minute).replace("'", "").split("+")[0])
                            player_appearances[player_id]['sub_minutes'].append(minute_num)
                        except:
                            pass

        # Calculate comprehensive player metrics
        for player_id, data in player_appearances.items():
            total_apps = data['starts'] + data['sub_appearances']
            start_rate = data['starts'] / total_matches if total_matches > 0 else 0

            # Determine player role
            if start_rate > 0.8:
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
            for sub_event in match.get('substitutions', []):
                player_id = sub_event['player_id']
                sa = substitution_analysis[player_id]

                sa['total_sub_apps'] += 1
                sa['results_when_subbed'][match['result']] += 1

                # Track substitution timing
                sub_minute = sub_event.get('sub_minute', 'Unknown')
                if sub_minute != 'Unknown':
                    try:
                        minute_num = int(str(sub_minute).replace("'", "").split("+")[0])
                        sa['sub_minutes'].append(minute_num)
                    except:
                        pass

                # Track performance as substitute
                rating = sub_event.get('rating', 0)
                if rating > 0:
                    sa['total_rating_as_sub'] += rating
                    sa['rating_count_as_sub'] += 1

        # Calculate averages
        for player_id, sa in substitution_analysis.items():
            if sa['sub_minutes']:
                sa['avg_sub_minute'] = round(np.mean(sa['sub_minutes']), 1)
            if sa['rating_count_as_sub'] > 0:
                sa['avg_rating_as_sub'] = round(sa['total_rating_as_sub'] / sa['rating_count_as_sub'], 2)

        team_data['substitution_analysis'] = substitution_analysis

    def _analyze_formation_performance(self, team_data):
        """Analyze detailed performance by formation"""
        for formation, data in team_data['formations'].items():
            formation_matches = [m for m in team_data['matches'] if m.get('formation') == formation]

            if formation_matches:
                # Calculate advanced stats
                avg_stats = {}
                stat_keys = ['expected_goals_xg', 'ball_possession', 'total_shots', 'shots_on_target',
                           'big_chances', 'accurate_passes', 'fouls_committed', 'corners']

                for key in stat_keys:
                    values = [m['stats'].get(key, 0) for m in formation_matches if key in m.get('stats', {})]
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
            if data['name']:  # Only include players with names
                player_roles[data['role']].append(data)

        for role, players in player_roles.items():
            if players:
                report.append(f"\n{role} ({len(players)} players):")
                for player in sorted(players, key=lambda x: x['start_rate'], reverse=True)[:15]:
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

                    report.append(f"  • {player['name']} ({player['primary_position']}): "
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

# Auto-load PKL data on startup
if not st.session_state.data_loaded:
    base_url = "https://github.com/sznajdr/cmpo/raw/refs/heads/main"
    if st.session_state.analyzer.load_optimized_data(base_url):
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
        st.error("Failed to load team analysis data")

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
            csv_url = "https://raw.githubusercontent.com/sznajdr/cmpo/refs/heads/main/fdmbl.csv"
            df = pd.read_csv(csv_url)
            processed_df, _, _ = preprocess_csv(df)
            st.session_state.csv_data = processed_df
        except:
            # Fallback sample data
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
                    max_value = int(st.session_state.csv_data['player_market_value'].max() / 1000000) if not pd.isna(st.session_state.csv_data['player_market_value'].max()) else 100
                    value_range = st.slider("Market Value (M€):", min_value, max_value, (min_value, max_value))
                else:
                    value_range = (0, 100)
            except:
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
            st.warning(f"Filter error: {e}")
        
        # Format market value for display
        display_df = filtered_df.copy()
        try:
            if 'player_market_value' in display_df.columns:
                display_df['market_value_formatted'] = display_df['player_market_value'].apply(
                    lambda x: f"€{x/1000000:.1f}M" if pd.notna(x) and x > 0 else "-"
                )
        except:
            pass
        
        # Display results
        st.write(f"Showing {len(filtered_df)} of {len(st.session_state.csv_data)} players")
        
        if not filtered_df.empty:
            # Select columns to display
            all_possible_columns = [
                'player_name', 'club', 'position', 'age', 
                'nationality', 'league_name', 'player_market_value', 
                'data_type', 'injury'
            ]
            
            available_columns = [col for col in all_possible_columns if col in display_df.columns]
            if not available_columns:
                available_columns = display_df.columns.tolist()[:6]
            
            # Rename columns for display
            column_names = {
                'player_name': 'Player', 'club': 'Club', 'position': 'Position',
                'age': 'Age', 'nationality': 'Nationality', 'league_name': 'League',
                'player_market_value': 'Market Value', 'data_type': 'Type', 'injury': 'Injury'
            }
            
            display_df_show = display_df[available_columns].copy()
            for old_name, new_name in column_names.items():
                if old_name in display_df_show.columns:
                    display_df_show = display_df_show.rename(columns={old_name: new_name})
            
            st.dataframe(display_df_show, use_container_width=True, hide_index=True)
            
            # Download button
            csv_download = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_download,
                file_name="player_data.csv",
                mime="text/csv"
            )
        else:
            st.info("No players match the selected filters")
