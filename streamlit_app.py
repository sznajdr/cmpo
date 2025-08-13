import streamlit as st
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="FotMob Team Tactical Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FotMobAnalyzer:
    def __init__(self):
        self.data = None
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

    def load_data_from_json(self, json_data):
        """Load and validate data from JSON"""
        try:
            self.data = json_data
            
            # Extract team names
            teams = set()
            valid_matches = 0
            
            for match in self.data:
                # Validate match structure
                if not self._validate_match_structure(match):
                    continue
                    
                home_team = self._safe_get(match, 'general.homeTeam.name')
                away_team = self._safe_get(match, 'general.awayTeam.name')
                
                if home_team and away_team:
                    teams.add(home_team)
                    teams.add(away_team)
                    valid_matches += 1
            
            self.team_names = sorted(list(teams))
            
            if valid_matches == 0:
                st.error("❌ No valid matches found in the uploaded data")
                return False
                
            st.success(f"✅ Loaded {valid_matches} valid matches from {len(teams)} teams")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return False

    def _validate_match_structure(self, match):
        """Validate that match has required structure"""
        required_paths = [
            'general.homeTeam.name',
            'general.awayTeam.name',
            'header.teams'
        ]
        
        for path in required_paths:
            if self._safe_get(match, path) is None:
                return False
        
        teams = self._safe_get(match, 'header.teams', [])
        return len(teams) >= 2

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
            'substitution_analysis': {}
        }

        # Extract all matches for this team
        for match in self.data:
            if not self._validate_match_structure(match):
                continue
                
            match_info = self._extract_team_match_info(match, team_name)
            if match_info:
                team_data['matches'].append(match_info)

        if not team_data['matches']:
            return None

        # Sort matches by date
        team_data['matches'].sort(key=lambda x: x.get('date', ''))

        # Analyze different aspects
        self._analyze_team_formations(team_data)
        self._analyze_player_rotations(team_data)
        self._analyze_formation_performance(team_data)
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

        # Get lineup info
        lineup_key = 'homeTeam' if is_home else 'awayTeam'
        lineup_info = self._safe_get(match, f'content.lineup.{lineup_key}', {})

        formation = lineup_info.get('formation')
        starters = lineup_info.get('starters', [])
        subs = lineup_info.get('subs', [])

        # Get player stats
        player_stats = self._safe_get(match, 'content.playerStats', {})

        # Extract match stats
        stats = self._extract_team_match_stats(match, is_home)

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
            'team_score': team_score,
            'opponent_score': opponent_score,
            'result': 'W' if team_score > opponent_score else 'D' if team_score == opponent_score else 'L',
            'stats': stats
        }

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

                    # Extract key metrics safely
                    player_data['rating'] = self._parse_numeric_string(
                        self._safe_get(top_stats, 'FotMob rating.stat.value', 0)
                    )
                    player_data['minutes'] = self._parse_numeric_string(
                        self._safe_get(top_stats, 'Minutes played.stat.value', 0)
                    )
                    player_data['goals'] = self._parse_numeric_string(
                        self._safe_get(top_stats, 'Goals.stat.value', 0)
                    )
                    player_data['assists'] = self._parse_numeric_string(
                        self._safe_get(top_stats, 'Assists.stat.value', 0)
                    )
                    player_data['xG'] = self._parse_numeric_string(
                        self._safe_get(top_stats, 'Expected goals (xG).stat.value', 0)
                    )
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
            'positions_played': defaultdict(int),
            'goals': 0,
            'assists': 0,
            'xG': 0.0,
            'total_minutes': 0,
            'total_rating': 0,
            'rating_count': 0
        })

        total_matches = len(team_data['matches'])

        for match in team_data['matches']:
            # Track starters
            for player in match.get('starters', []):
                player_id = player['id']
                data = player_appearances[player_id]

                data['name'] = player['name']
                data['starts'] += 1
                data['positions_played'][player.get('position', '')] += 1
                data['goals'] += player.get('goals', 0)
                data['assists'] += player.get('assists', 0)
                data['xG'] += player.get('xG', 0.0)
                data['total_minutes'] += player.get('minutes', 0)

                if player.get('rating', 0) > 0:
                    data['total_rating'] += player['rating']
                    data['rating_count'] += 1

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

            team_data['player_pool'][player_id] = {
                'name': data['name'],
                'total_appearances': total_apps,
                'starts': data['starts'],
                'sub_appearances': data['sub_appearances'],
                'start_rate': start_rate * 100,
                'role': role,
                'primary_position': max(data['positions_played'], key=data['positions_played'].get) if data['positions_played'] else '',
                'avg_rating': data['total_rating'] / data['rating_count'] if data['rating_count'] > 0 else 0,
                'goals': data['goals'],
                'assists': data['assists'],
                'xG': data['xG'],
                'total_minutes': data['total_minutes'],
                'minutes_per_game': data['total_minutes'] / total_apps if total_apps > 0 else 0
            }

    def _analyze_substitution_patterns(self, team_data):
        """Analyze substitution patterns"""
        # Simple substitution analysis based on available data
        sub_players = [p for p in team_data['player_pool'].values() if p['sub_appearances'] > 0]
        team_data['substitution_analysis'] = {
            'total_substitute_players': len(sub_players),
            'most_used_sub': max(sub_players, key=lambda x: x['sub_appearances']) if sub_players else None
        }

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

    def display_team_analysis(self, team_name):
        """Display comprehensive team analysis in Streamlit"""
        team_data = self.analyze_team_tactical_profile(team_name)
        
        if not team_data:
            st.error(f"❌ No data found for {team_name}")
            return

        st.header(f"🏆 {team_name.upper()} - TACTICAL ANALYSIS")
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Squad Analysis", 
            "📊 Formation Performance", 
            "📈 Team Statistics",
            "🧠 Key Insights"
        ])

        with tab1:
            self._display_squad_analysis(team_data)
        
        with tab2:
            self._display_formation_analysis(team_data)
        
        with tab3:
            self._display_team_statistics(team_data)
        
        with tab4:
            self._display_key_insights(team_data)

    def _display_squad_analysis(self, team_data):
        """Display squad analysis"""
        st.subheader("👥 Squad Rotation Analysis")
        
        # Group players by role
        player_roles = defaultdict(list)
        for player_id, data in team_data['player_pool'].items():
            if data['name']:
                player_roles[data['role']].append(data)

        for role, players in player_roles.items():
            if players:
                st.write(f"**{role} ({len(players)} players):**")
                
                # Create DataFrame for better display
                player_data = []
                for player in sorted(players, key=lambda x: x['start_rate'], reverse=True):
                    player_data.append({
                        'Name': player['name'],
                        'Position': player['primary_position'],
                        'Apps': f"{player['starts']}S + {player['sub_appearances']}Sub",
                        'Start Rate': f"{player['start_rate']:.0f}%",
                        'Goals': player['goals'],
                        'Assists': player['assists'],
                        'xG': f"{player['xG']:.1f}",
                        'Rating': f"{player['avg_rating']:.1f}" if player['avg_rating'] > 0 else "N/A",
                        'Minutes/Game': f"{player['minutes_per_game']:.0f}" if player['minutes_per_game'] > 0 else "0"
                    })
                
                if player_data:
                    df = pd.DataFrame(player_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)

    def _display_formation_analysis(self, team_data):
        """Display formation performance analysis"""
        st.subheader("🏟️ Formation Performance")
        
        if not team_data['formations']:
            st.warning("No formation data available")
            return
        
        sorted_formations = sorted(
            team_data['formations'].items(),
            key=lambda x: x[1]['usage_count'],
            reverse=True
        )

        for formation, data in sorted_formations:
            with st.expander(f"📋 {formation} Formation ({data['usage_count']} matches)", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Win Rate", f"{data['win_rate']:.1f}%")
                
                with col2:
                    st.metric("Points/Game", f"{data['points_per_game']:.2f}")
                
                with col3:
                    st.metric("Goals For", f"{data['goals_for_avg']:.1f}/game")
                
                with col4:
                    st.metric("Goals Against", f"{data['goals_against_avg']:.1f}/game")
                
                # Advanced stats if available
                perf_data = team_data['performance_by_formation'].get(formation, {})
                if perf_data and perf_data.get('avg_possession', 0) > 0:
                    st.write("**Advanced Statistics:**")
                    adv_col1, adv_col2 = st.columns(2)
                    
                    with adv_col1:
                        st.write(f"• Possession: {perf_data.get('avg_possession', 0):.1f}%")
                        st.write(f"• Shots: {perf_data.get('avg_shots', 0):.1f}/game")
                    
                    with adv_col2:
                        st.write(f"• Big Chances: {perf_data.get('avg_big_chances', 0):.1f}/game")
                        st.write(f"• Style: {perf_data.get('style_profile', 'Unknown')}")

    def _display_team_statistics(self, team_data):
        """Display team statistics"""
        st.subheader("📊 Overall Team Statistics")
        
        team_matches = team_data['matches']
        
        if not team_matches:
            st.warning("No match data available")
            return
        
        # Basic stats
        total_matches = len(team_matches)
        total_goals_for = sum(match['team_score'] for match in team_matches)
        total_goals_against = sum(match['opponent_score'] for match in team_matches)
        
        # Calculate record
        results = [match['result'] for match in team_matches]
        wins = results.count('W')
        draws = results.count('D')
        losses = results.count('L')
        points = wins * 3 + draws

        # Display key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Matches", total_matches)
        
        with col2:
            st.metric("Wins", wins)
        
        with col3:
            st.metric("Draws", draws)
        
        with col4:
            st.metric("Losses", losses)
        
        with col5:
            st.metric("Points", points)

        # Goals
        st.write("**Goal Statistics:**")
        goal_col1, goal_col2, goal_col3 = st.columns(3)
        
        with goal_col1:
            st.metric("Goals For", total_goals_for, f"{total_goals_for/total_matches:.2f}/game")
        
        with goal_col2:
            st.metric("Goals Against", total_goals_against, f"{total_goals_against/total_matches:.2f}/game")
        
        with goal_col3:
            st.metric("Goal Difference", total_goals_for - total_goals_against)

        # Home vs Away if available
        home_matches = [m for m in team_matches if m['is_home']]
        away_matches = [m for m in team_matches if not m['is_home']]

        if home_matches and away_matches:
            st.write("**Home vs Away Performance:**")
            
            home_results = [m['result'] for m in home_matches]
            away_results = [m['result'] for m in away_matches]
            
            home_record = f"{home_results.count('W')}W-{home_results.count('D')}D-{home_results.count('L')}L"
            away_record = f"{away_results.count('W')}W-{away_results.count('D')}D-{away_results.count('L')}L"
            
            home_col, away_col = st.columns(2)
            
            with home_col:
                st.info(f"🏠 **Home:** {home_record}")
            
            with away_col:
                st.info(f"✈️ **Away:** {away_record}")

    def _display_key_insights(self, team_data):
        """Display key tactical insights"""
        st.subheader("🔍 Key Tactical Insights")
        
        # Best formation
        if team_data['formations']:
            best_formation = max(
                [(f, d) for f, d in team_data['formations'].items() if d['usage_count'] >= 2],
                key=lambda x: x[1]['win_rate'],
                default=(None, None)
            )
            
            if best_formation[0]:
                st.success(f"🏆 **Most Successful Formation:** {best_formation[0]} ({best_formation[1]['win_rate']:.1f}% win rate)")

        # Top performers
        key_players = [p for p in team_data['player_pool'].values() if p['start_rate'] > 60 and p['starts'] >= 3]
        if key_players:
            top_performer = max(key_players, key=lambda x: x['avg_rating'] if x['avg_rating'] > 0 else 0)
            if top_performer['avg_rating'] > 0:
                st.info(f"⭐ **Top Performer:** {top_performer['name']} ({top_performer['avg_rating']:.1f} avg rating)")

        # Most used formation
        if team_data['formations']:
            most_used = max(team_data['formations'].items(), key=lambda x: x[1]['usage_count'])
            st.info(f"📊 **Preferred Formation:** {most_used[0]} (used {most_used[1]['usage_count']} times)")

        # Squad depth analysis
        total_players_used = len([p for p in team_data['player_pool'].values() if p['total_appearances'] > 0])
        regular_starters = len([p for p in team_data['player_pool'].values() if p['start_rate'] > 50])
        
        st.write("**Squad Depth Analysis:**")
        depth_col1, depth_col2 = st.columns(2)
        
        with depth_col1:
            st.write(f"• Total Players Used: {total_players_used}")
            st.write(f"• Regular Starters: {regular_starters}")
        
        with depth_col2:
            rotation_rate = ((total_players_used - regular_starters) / total_players_used * 100) if total_players_used > 0 else 0
            st.write(f"• Squad Rotation Rate: {rotation_rate:.1f}%")
            
            if rotation_rate > 60:
                st.write("📈 High rotation squad")
            elif rotation_rate > 40:
                st.write("📊 Moderate rotation")
            else:
                st.write("📉 Stable starting XI")


def main():
    st.title("⚽ FotMob Team Tactical Analysis")
    st.markdown("**Upload your FotMob JSON data for comprehensive tactical analysis**")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = FotMobAnalyzer()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'team_names' not in st.session_state:
        st.session_state.team_names = []

    analyzer = st.session_state.analyzer
    
    # Sidebar for file upload and instructions
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # Instructions
        with st.expander("ℹ️ How to get FotMob data"):
            st.markdown("""
            **Option 1: Use the Jupyter Notebook locally**
            1. Download and run the original notebook on your computer
            2. Configure the x-mas header from FotMob
            3. Scrape the data and download the JSON file
            4. Upload the JSON file here
            
            **Option 2: Manual API collection**
            1. Go to fotmob.com and inspect network requests
            2. Find match detail API calls
            3. Copy the JSON responses
            4. Create a JSON array and upload
            
            **Data Format Expected:**
            - Array of match objects with team info, lineups, and statistics
            - Each match should have `general`, `header`, `content` sections
            """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a JSON file", 
            type="json",
            help="Upload FotMob match data in JSON format"
        )
        
        if uploaded_file is not None:
            try:
                # Load JSON data
                json_data = json.load(uploaded_file)
                
                # Validate it's a list
                if not isinstance(json_data, list):
                    st.error("❌ JSON file should contain an array of matches")
                elif len(json_data) == 0:
                    st.error("❌ JSON file is empty")
                else:
                    # Load data into analyzer
                    if analyzer.load_data_from_json(json_data):
                        st.session_state.team_names = analyzer.team_names
                        st.session_state.data_loaded = True
                        st.rerun()
                    else:
                        st.session_state.data_loaded = False
                        
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file format")
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        
        # Sample data option
        st.subheader("🧪 Try Sample Data")
        if st.button("Load Sample Data"):
            # Create sample data structure
            sample_data = [
                {
                    "general": {
                        "matchId": "sample001",
                        "homeTeam": {"name": "Sample FC", "score": 2},
                        "awayTeam": {"name": "Example United", "score": 1},
                        "matchTimeUTCDate": "2024-01-15",
                        "leagueName": "Sample League",
                        "matchRound": "1"
                    },
                    "header": {
                        "teams": [
                            {"score": 2},
                            {"score": 1}
                        ]
                    },
                    "content": {
                        "lineup": {
                            "homeTeam": {
                                "formation": "4-3-3",
                                "starters": [
                                    {
                                        "id": "p1",
                                        "name": "John Goalkeeper", 
                                        "positionId": 1,
                                        "positionLabel": {"label": "GK"},
                                        "shirtNumber": 1
                                    },
                                    {
                                        "id": "p2", 
                                        "name": "Mike Defender",
                                        "positionId": 32,
                                        "positionLabel": {"label": "RB"},
                                        "shirtNumber": 2
                                    }
                                ],
                                "subs": [
                                    {
                                        "id": "p3",
                                        "name": "Sub Player",
                                        "positionId": 101,
                                        "positionLabel": {"label": "ST"},
                                        "shirtNumber": 9
                                    }
                                ]
                            },
                            "awayTeam": {
                                "formation": "4-4-2",
                                "starters": [
                                    {
                                        "id": "p4",
                                        "name": "Away Keeper",
                                        "positionId": 1,
                                        "positionLabel": {"label": "GK"},
                                        "shirtNumber": 1
                                    }
                                ],
                                "subs": []
                            }
                        },
                        "playerStats": {},
                        "stats": {
                            "Periods": {
                                "All": {
                                    "stats": [
                                        {
                                            "key": "top_stats",
                                            "stats": [
                                                {
                                                    "title": "Ball possession",
                                                    "stats": ["60%", "40%"]
                                                },
                                                {
                                                    "title": "Total shots",
                                                    "stats": ["12", "8"]
                                                }
                                            ]
                                        }
                                    ]
                                }
                            }
                        }
                    }
                }
            ]
            
            if analyzer.load_data_from_json(sample_data):
                st.session_state.team_names = analyzer.team_names
                st.session_state.data_loaded = True
                st.success("✅ Sample data loaded!")
                st.rerun()

    # Main content area
    if st.session_state.data_loaded and st.session_state.team_names:
        st.success(f"📊 Data loaded successfully! Found {len(st.session_state.team_names)} teams")
        
        # Team selection
        st.subheader("🎯 Select Team for Analysis")
        
        # Create two columns for team selection
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_team = st.selectbox(
                "Choose a team:",
                options=["Select a team..."] + st.session_state.team_names,
                index=0
            )
        
        with col2:
            st.write("") # Spacer
            if st.button("🔄 Refresh Analysis"):
                if selected_team != "Select a team...":
                    st.rerun()
        
        # Display analysis
        if selected_team != "Select a team...":
            with st.spinner(f"Analyzing {selected_team}..."):
                analyzer.display_team_analysis(selected_team)
            
            # Download options
            st.divider()
            st.subheader("💾 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create summary for selected team
                team_data = analyzer.analyze_team_tactical_profile(selected_team)
                if team_data:
                    summary = {
                        "team_name": selected_team,
                        "total_matches": len(team_data['matches']),
                        "formations_used": list(team_data['formations'].keys()),
                        "key_players": [
                            p['name'] for p in team_data['player_pool'].values() 
                            if p['start_rate'] > 70
                        ]
                    }
                    
                    summary_json = json.dumps(summary, indent=2)
                    st.download_button(
                        label="📥 Download Team Summary",
                        data=summary_json,
                        file_name=f"{selected_team.replace(' ', '_')}_summary.json",
                        mime="application/json"
                    )
            
            with col2:
                # Create detailed CSV export
                if team_data and team_data['player_pool']:
                    player_df_data = []
                    for player_id, player in team_data['player_pool'].items():
                        player_df_data.append({
                            'Player': player['name'],
                            'Position': player['primary_position'],
                            'Starts': player['starts'],
                            'Sub_Apps': player['sub_appearances'],
                            'Start_Rate_%': round(player['start_rate'], 1),
                            'Goals': player['goals'],
                            'Assists': player['assists'],
                            'xG': round(player['xG'], 2),
                            'Avg_Rating': round(player['avg_rating'], 2) if player['avg_rating'] > 0 else 0,
                            'Minutes_Per_Game': round(player['minutes_per_game'], 1),
                            'Role': player['role']
                        })
                    
                    player_df = pd.DataFrame(player_df_data)
                    csv = player_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download Player Data CSV",
                        data=csv,
                        file_name=f"{selected_team.replace(' ', '_')}_players.csv",
                        mime="text/csv"
                    )
        
        # Team comparison option
        st.divider()
        st.subheader("⚖️ Quick Team Comparison")
        
        if len(st.session_state.team_names) >= 2:
            compare_col1, compare_col2 = st.columns(2)
            
            with compare_col1:
                team1 = st.selectbox("Team 1:", st.session_state.team_names, key="team1")
            
            with compare_col2:
                team2 = st.selectbox("Team 2:", st.session_state.team_names, key="team2")
            
            if st.button("🔬 Compare Teams"):
                if team1 != team2:
                    team1_data = analyzer.analyze_team_tactical_profile(team1)
                    team2_data = analyzer.analyze_team_tactical_profile(team2)
                    
                    if team1_data and team2_data:
                        comp_col1, comp_col2 = st.columns(2)
                        
                        with comp_col1:
                            st.write(f"**{team1}**")
                            st.write(f"Matches: {len(team1_data['matches'])}")
                            if team1_data['formations']:
                                most_used = max(team1_data['formations'].items(), key=lambda x: x[1]['usage_count'])
                                st.write(f"Preferred Formation: {most_used[0]}")
                            
                            key_players_1 = [p['name'] for p in team1_data['player_pool'].values() if p['start_rate'] > 70]
                            st.write(f"Key Players: {len(key_players_1)}")
                        
                        with comp_col2:
                            st.write(f"**{team2}**")
                            st.write(f"Matches: {len(team2_data['matches'])}")
                            if team2_data['formations']:
                                most_used = max(team2_data['formations'].items(), key=lambda x: x[1]['usage_count'])
                                st.write(f"Preferred Formation: {most_used[0]}")
                            
                            key_players_2 = [p['name'] for p in team2_data['player_pool'].values() if p['start_rate'] > 70]
                            st.write(f"Key Players: {len(key_players_2)}")
                else:
                    st.warning("Please select two different teams")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to FotMob Team Tactical Analysis
        
        This application provides comprehensive tactical analysis of football teams using FotMob data.
        
        ### 📊 **What you can analyze:**
        
        #### 👥 **Squad Analysis**
        - Player rotation patterns and roles
        - Starting XI vs substitute usage
        - Performance metrics for each player
        - Position-based analysis
        
        #### 🏟️ **Formation Analysis**
        - Formation usage frequency
        - Win rates by formation
        - Tactical style identification
        - Performance metrics per formation
        
        #### 📈 **Team Statistics**
        - Overall team performance
        - Home vs away records
        - Goal statistics and averages
        - Match outcome analysis
        
        #### 🔍 **Key Insights**
        - Best performing formations
        - Top performing players
        - Squad depth analysis
        - Tactical recommendations
        
        ---
        
        ### 🛠️ **How to get started:**
        
        1. **📁 Upload Data**: Use the sidebar to upload your FotMob JSON data
        2. **🎯 Select Team**: Choose a team from the dropdown
        3. **📊 Analyze**: Explore the different analysis tabs
        4. **💾 Export**: Download summaries and detailed data
        
        ### 📋 **Data Requirements:**
        
        The app expects JSON data in FotMob format containing:
        - Match information (teams, scores, dates)
        - Team lineups and formations
        - Player statistics and performance data
        - Match statistics (possession, shots, etc.)
        
        ### 🧪 **Try it out:**
        
        Click "Load Sample Data" in the sidebar to see how the analysis works with sample data!
        
        ---
        
        **💡 Tip:** For best results, upload data from multiple matches of the same team to get comprehensive tactical insights.
        """)
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide"):
            st.markdown("""
            **Method 1: Use Jupyter Notebook (Recommended)**
            1. Download the original Jupyter notebook
            2. Run it locally with proper x-mas header
            3. Export the JSON file
            4. Upload here for analysis
            
            **Method 2: Manual Collection**
            1. Visit fotmob.com
            2. Open browser developer tools
            3. Navigate to a match page
            4. Find the match details API call in Network tab
            5. Copy the JSON response
            6. Create an array of match objects and upload
            
            **Method 3: Sample Data**
            1. Click "Load Sample Data" in sidebar
            2. Explore the interface with demo data
            3. Understand the expected data format
            """)


if __name__ == "__main__":
    main()
