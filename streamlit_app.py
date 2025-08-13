import streamlit as st
import json
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from collections import defaultdict, Counter
import os
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="FotMob Team Tactical Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitFotMobAnalyzer:
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
        
        # Initialize session state
        if 'scraped_data' not in st.session_state:
            st.session_state.scraped_data = None
        if 'team_names' not in st.session_state:
            st.session_state.team_names = []
        if 'scraping_complete' not in st.session_state:
            st.session_state.scraping_complete = False
        if 'headers' not in st.session_state:
            st.session_state.headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "accept": "application/json, text/plain, */*",
                "x-mas": "PLACEHOLDER_HEADER_PLEASE_UPDATE"  # User needs to update this
            }

    def scrape_fotmob_data(self, league_id, slug, season, num_rounds, headers):
        """Scrape FotMob data for given parameters"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Get match URLs
            status_text.text("🔍 Fetching match URLs...")
            match_urls = []
            
            base_url = f"https://www.fotmob.com/leagues/{league_id}/matches/{slug}"
            url = f"{base_url}?season={season}"
            
            for round_num in range(num_rounds):
                progress = (round_num + 1) / (num_rounds + 1)
                progress_bar.progress(progress)
                status_text.text(f"📥 Fetching round {round_num + 1}/{num_rounds}")
                
                round_url = f"{url}&group=by-round&round={round_num}"
                
                # For this example, we'll simulate the URL structure
                # In a real implementation, you'd need to use Selenium or similar
                # to get the actual match URLs from the JavaScript-rendered page
                
                # Simulated match URLs - replace with actual scraping logic
                for match_idx in range(10):  # Assume 10 matches per round
                    match_urls.append(f"https://www.fotmob.com/matches/{league_id}_{round_num}_{match_idx}#{league_id}{round_num}{match_idx:02d}")
            
            # Step 2: Fetch match details
            status_text.text("⚽ Fetching match details...")
            match_data = []
            failed_matches = []
            
            api_base = "https://www.fotmob.com/api/data/matchDetails?matchId="
            
            for i, url in enumerate(match_urls):
                progress = 0.5 + (i / len(match_urls)) * 0.5
                progress_bar.progress(progress)
                status_text.text(f"🔄 Processing match {i+1}/{len(match_urls)}")
                
                # Extract match ID from URL
                if '#' in url:
                    match_id = url.split('#')[-1].strip()
                    
                    try:
                        response = requests.get(f"{api_base}{match_id}", headers=headers, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            if 'content' in data:
                                match_data.append(data)
                            else:
                                failed_matches.append((match_id, "No content in response"))
                        elif response.status_code == 403:
                            st.error("❌ Authentication failed - x-mas header expired or invalid")
                            return None, []
                        else:
                            failed_matches.append((match_id, f"Status {response.status_code}"))
                    except Exception as e:
                        failed_matches.append((match_id, str(e)))
                        
                    time.sleep(0.5)  # Rate limiting
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Scraping complete! {len(match_data)} matches collected, {len(failed_matches)} failed")
            
            return match_data, failed_matches
            
        except Exception as e:
            st.error(f"❌ Error during scraping: {e}")
            return None, []

    def load_data_from_json(self, json_data):
        """Load data from JSON"""
        try:
            self.data = json_data
            
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
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
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

    def display_team_analysis(self, team_name):
        """Display comprehensive team analysis in Streamlit"""
        team_data = self.analyze_team_tactical_profile(team_name)
        
        if not team_data:
            st.error(f"❌ No data found for {team_name}")
            return

        st.header(f"🏆 {team_name.upper()} - COMPREHENSIVE TACTICAL ANALYSIS")
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📄 Squad Rotation", 
            "📊 Formation Analysis", 
            "🔄 Substitutions", 
            "📈 Team Statistics",
            "🧠 Tactical Insights"
        ])

        with tab1:
            self._display_squad_rotation(team_data)
        
        with tab2:
            self._display_formation_analysis(team_data)
        
        with tab3:
            self._display_substitution_analysis(team_data)
        
        with tab4:
            self._display_team_statistics(team_data)
        
        with tab5:
            self._display_tactical_insights(team_data)

    def _display_squad_rotation(self, team_data):
        """Display squad rotation analysis"""
        st.subheader("🔄 SQUAD ROTATION ANALYSIS")
        
        # Group players by role
        player_roles = defaultdict(list)
        for player_id, data in team_data['player_pool'].items():
            if data['name']:  # Only include players with names
                player_roles[data['role']].append(data)

        for role, players in player_roles.items():
            if players:
                st.write(f"**{role} ({len(players)} players):**")
                
                # Create DataFrame for better display
                player_data = []
                for player in sorted(players, key=lambda x: x['start_rate'], reverse=True)[:15]:
                    player_data.append({
                        'Name': player['name'],
                        'Position': player['primary_position'],
                        'Starts': f"{player['starts']}S+{player['sub_appearances']}Sub",
                        'Start Rate': f"{player['start_rate']:.0f}%",
                        'Goals+Assists': f"{player['goals']}G+{player['assists']}A" if player['goals'] > 0 or player['assists'] > 0 else "",
                        'Rating': f"{player['avg_rating']:.1f}⭐" if player['avg_rating'] > 0 else "",
                        'Minutes/Game': f"{player['minutes_per_game']:.0f}" if player['minutes_per_game'] > 0 else "",
                        'Form': f"{player['recent_form_avg']:.1f}" if player['recent_form_avg'] > 0 else ""
                    })
                
                if player_data:
                    df = pd.DataFrame(player_data)
                    st.dataframe(df, use_container_width=True)
                st.write("---")

    def _display_formation_analysis(self, team_data):
        """Display formation performance analysis"""
        st.subheader("📊 DETAILED FORMATION PERFORMANCE")
        
        sorted_formations = sorted(
            team_data['formations'].items(),
            key=lambda x: x[1]['usage_count'],
            reverse=True
        )

        for formation, data in sorted_formations:
            with st.expander(f"🏟️ {formation} Formation ({data['usage_count']} matches)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Win Rate", f"{data['win_rate']:.1f}%")
                    st.metric("Points per Game", f"{data['points_per_game']:.2f}")
                    st.metric("Goals For/Game", f"{data['goals_for_avg']:.1f}")
                    st.metric("Goals Against/Game", f"{data['goals_against_avg']:.1f}")
                
                with col2:
                    perf_data = team_data['performance_by_formation'].get(formation, {})
                    if perf_data:
                        st.metric("Average Possession", f"{perf_data.get('avg_possession', 0):.1f}%")
                        st.metric("Shots per Game", f"{perf_data.get('avg_shots', 0):.1f}")
                        st.metric("Big Chances per Game", f"{perf_data.get('avg_big_chances', 0):.1f}")
                        st.write(f"**Style:** {perf_data.get('style_profile', 'Unknown')}")

    def _display_substitution_analysis(self, team_data):
        """Display substitution impact analysis"""
        st.subheader("🔄 SUBSTITUTION IMPACT ANALYSIS")
        
        # Get players with significant substitute appearances
        sub_players = []
        for player_id, player_data in team_data['player_pool'].items():
            if player_data['sub_appearances'] >= 2:
                sub_info = {
                    'Name': player_data['name'],
                    'Position': player_data['primary_position'],
                    'Sub Apps': player_data['sub_appearances'],
                    'Avg Sub Minute': f"{player_data['avg_sub_minute']:.0f}'" if player_data['avg_sub_minute'] > 0 else "Various",
                    'Goals': player_data['goals'],
                    'Assists': player_data['assists'],
                    'xG': f"{player_data['xG']:.1f}",
                    'Rating': f"{player_data['avg_rating']:.1f}" if player_data['avg_rating'] > 0 else "",
                    'Role': "🟠 Super-Sub" if player_data['start_rate'] < 50 else "🟡 Rotation"
                }
                sub_players.append(sub_info)

        if sub_players:
            df_subs = pd.DataFrame(sub_players)
            st.dataframe(df_subs, use_container_width=True)
        else:
            st.warning("⚠️ No regular substitute players found")

    def _display_team_statistics(self, team_data):
        """Display comprehensive team statistics"""
        st.subheader("📈 COMPREHENSIVE TEAM STATISTICS")
        
        team_matches = team_data['matches']
        
        if team_matches:
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
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Matches", total_matches)
                st.metric("Wins", wins)
            
            with col2:
                st.metric("Draws", draws)
                st.metric("Losses", losses)
            
            with col3:
                st.metric("Points", points)
                st.metric("Goals For", total_goals_for)
            
            with col4:
                st.metric("Goals Against", total_goals_against)
                st.metric("Goal Difference", total_goals_for - total_goals_against)

            # Calculate averages from match stats
            match_stats_keys = ['expected_goals_xg', 'ball_possession', 'total_shots', 'shots_on_target',
                              'big_chances', 'accurate_passes', 'fouls_committed', 'corners']

            team_averages = {}
            opponent_averages = {}

            for key in match_stats_keys:
                team_values = [match['stats'].get(key, 0) for match in team_matches if key in match.get('stats', {})]
                opponent_values = [match['stats'].get(f'opponent_{key}', 0) for match in team_matches if f'opponent_{key}' in match.get('stats', {})]

                team_averages[key] = np.mean(team_values) if team_values else 0
                opponent_averages[key] = np.mean(opponent_values) if opponent_values else 0

            # Display advanced stats
            st.subheader("Advanced Statistics")
            
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                st.write("**Team Averages:**")
                if team_averages.get('ball_possession', 0) > 0:
                    st.write(f"• Possession: {team_averages['ball_possession']:.1f}%")
                if team_averages.get('total_shots', 0) > 0:
                    st.write(f"• Shots per Game: {team_averages['total_shots']:.1f}")
                if team_averages.get('shots_on_target', 0) > 0:
                    st.write(f"• Shots on Target: {team_averages['shots_on_target']:.1f}")
                if team_averages.get('big_chances', 0) > 0:
                    st.write(f"• Big Chances: {team_averages['big_chances']:.1f}")
            
            with adv_col2:
                st.write("**Opponent Averages:**")
                if opponent_averages.get('ball_possession', 0) > 0:
                    st.write(f"• Possession: {opponent_averages['ball_possession']:.1f}%")
                if opponent_averages.get('total_shots', 0) > 0:
                    st.write(f"• Shots per Game: {opponent_averages['total_shots']:.1f}")
                if opponent_averages.get('shots_on_target', 0) > 0:
                    st.write(f"• Shots on Target: {opponent_averages['shots_on_target']:.1f}")
                if opponent_averages.get('big_chances', 0) > 0:
                    st.write(f"• Big Chances: {opponent_averages['big_chances']:.1f}")

            # Home vs Away performance
            home_matches = [m for m in team_matches if m['is_home']]
            away_matches = [m for m in team_matches if not m['is_home']]

            if home_matches and away_matches:
                st.subheader("Home vs Away Performance")
                
                home_results = [m['result'] for m in home_matches]
                away_results = [m['result'] for m in away_matches]
                
                home_record = f"{home_results.count('W')}W-{home_results.count('D')}D-{home_results.count('L')}L"
                away_record = f"{away_results.count('W')}W-{away_results.count('D')}D-{away_results.count('L')}L"
                
                home_goals = sum(m['team_score'] for m in home_matches) / len(home_matches)
                away_goals = sum(m['team_score'] for m in away_matches) / len(away_matches)
                
                home_col, away_col = st.columns(2)
                
                with home_col:
                    st.write(f"**🏠 Home: {home_record}**")
                    st.write(f"Goals per Game: {home_goals:.2f}")
                
                with away_col:
                    st.write(f"**✈️ Away: {away_record}**")
                    st.write(f"Goals per Game: {away_goals:.2f}")

    def _display_tactical_insights(self, team_data):
        """Display tactical insights"""
        st.subheader("🧠 TACTICAL INSIGHTS")
        
        # Best formation
        best_formation = None
        best_win_rate = 0
        for formation, data in team_data['formations'].items():
            if data['usage_count'] >= 3 and data['win_rate'] > best_win_rate:
                best_win_rate = data['win_rate']
                best_formation = formation

        if best_formation:
            st.success(f"🏆 Most Successful Formation: **{best_formation}** ({best_win_rate:.1f}% win rate)")

        # Key players identification
        key_starters = [p for p in team_data['player_pool'].values() if p['start_rate'] > 70 and p['starts'] >= 5]
        if key_starters:
            top_performer = max(key_starters, key=lambda x: x['avg_rating'])
            st.info(f"⭐ Top Performer: **{top_performer['name']}** ({top_performer['avg_rating']:.1f} avg rating)")

        # Super sub identification
        super_subs = [p for p in team_data['player_pool'].values() if p['sub_appearances'] >= 3 and p['start_rate'] < 50]
        if super_subs:
            best_sub = max(super_subs, key=lambda x: x['xG'])
            st.info(f"🟠 Best Super-Sub: **{best_sub['name']}** ({best_sub['xG']:.1f} xG)")

        # Most used formation
        most_used = max(team_data['formations'].items(), key=lambda x: x[1]['usage_count'])
        st.info(f"📊 Preferred Formation: **{most_used[0]}** ({most_used[1]['usage_count']} times)")


# Main Streamlit App
def main():
    st.title("⚽ FotMob Team Tactical Analysis")
    st.markdown("Comprehensive tactical analysis tool for football teams using FotMob data")
    
    analyzer = StreamlitFotMobAnalyzer()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Headers configuration
        st.subheader("API Headers")
        with st.expander("⚠️ Update x-mas Header"):
            st.warning("You need to update the x-mas header from FotMob API")
            st.markdown("""
            **How to get x-mas header:**
            1. Go to fotmob.com
            2. Open Developer Tools (F12)
            3. Go to Network tab
            4. Look for API requests to fotmob.com/api
            5. Copy the 'x-mas' header value
            """)
            
            xmas_header = st.text_input(
                "x-mas Header Value:", 
                value=st.session_state.headers.get("x-mas", ""),
                type="password"
            )
            
            if st.button("Update Header"):
                st.session_state.headers["x-mas"] = xmas_header
                st.success("Header updated!")
        
        st.subheader("League Configuration")
        
        # League inputs
        league_id = st.text_input("League ID:", value="146", help="e.g., 146 for 2. Bundesliga")
        slug = st.text_input("League Slug:", value="2.-bundesliga", help="e.g., 2.-bundesliga")
        season = st.text_input("Season:", value="2025-2026", help="e.g., 2025-2026")
        num_rounds = st.number_input("Number of Rounds:", min_value=1, max_value=38, value=5)
        
        # Scraping button
        if st.button("🚀 Start Scraping", type="primary"):
            if not xmas_header or xmas_header == "PLACEHOLDER_HEADER_PLEASE_UPDATE":
                st.error("❌ Please update the x-mas header first!")
            else:
                st.session_state.headers["x-mas"] = xmas_header
                with st.spinner("Scraping data..."):
                    match_data, failed_matches = analyzer.scrape_fotmob_data(
                        league_id, slug, season, num_rounds, st.session_state.headers
                    )
                    
                    if match_data:
                        st.session_state.scraped_data = match_data
                        
                        # Load data into analyzer
                        if analyzer.load_data_from_json(match_data):
                            st.session_state.team_names = analyzer.team_names
                            st.session_state.scraping_complete = True
                            st.success(f"✅ Successfully scraped {len(match_data)} matches!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to process scraped data")
                    else:
                        st.error("❌ Failed to scrape data")
        
        # File upload option
        st.subheader("📁 Or Upload JSON File")
        uploaded_file = st.file_uploader("Choose a JSON file", type="json")
        
        if uploaded_file is not None:
            try:
                json_data = json.load(uploaded_file)
                st.session_state.scraped_data = json_data
                
                if analyzer.load_data_from_json(json_data):
                    st.session_state.team_names = analyzer.team_names
                    st.session_state.scraping_complete = True
                    st.success(f"✅ Successfully loaded {len(json_data)} matches!")
                    st.rerun()
                else:
                    st.error("❌ Failed to process uploaded data")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")

    # Main content area
    if st.session_state.scraping_complete and st.session_state.scraped_data:
        # Load data into analyzer if not already loaded
        if not analyzer.data:
            analyzer.load_data_from_json(st.session_state.scraped_data)
        
        st.success(f"📊 Data loaded: {len(st.session_state.scraped_data)} matches, {len(st.session_state.team_names)} teams")
        
        # Team selection
        st.subheader("🎯 Select Team for Analysis")
        
        selected_team = st.selectbox(
            "Choose a team:",
            options=["Select a team..."] + st.session_state.team_names,
            index=0
        )
        
        if selected_team != "Select a team...":
            analyzer.display_team_analysis(selected_team)
            
            # Download options
            st.subheader("💾 Download Data")
            col1, col2 = st.columns(2)
            
            with col1:
                # Download raw JSON
                json_str = json.dumps(st.session_state.scraped_data, indent=2)
                st.download_button(
                    label="📥 Download Raw JSON",
                    data=json_str,
                    file_name=f"fotmob_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Download summary CSV
                summary_data = []
                for match in st.session_state.scraped_data:
                    general = match.get('general', {})
                    summary_data.append({
                        'match_id': general.get('matchId'),
                        'home_team': general.get('homeTeam', {}).get('name'),
                        'away_team': general.get('awayTeam', {}).get('name'),
                        'home_score': general.get('homeTeam', {}).get('score'),
                        'away_score': general.get('awayTeam', {}).get('score'),
                        'date': general.get('matchTimeUTCDate'),
                        'league': general.get('leagueName'),
                        'round': general.get('matchRound')
                    })
                
                df_summary = pd.DataFrame(summary_data)
                csv = df_summary.to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary CSV",
                    data=csv,
                    file_name=f"fotmob_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to FotMob Team Tactical Analysis
        
        This application allows you to:
        
        ### 📊 **Scrape Match Data**
        - Enter league ID, slug, and season information
        - Automatically fetch match data from FotMob API
        - Process comprehensive match statistics
        
        ### 🧠 **Analyze Team Tactics**
        - Squad rotation patterns with detailed player statistics
        - Formation performance with advanced metrics
        - Substitution impact analysis with timing patterns
        - Comprehensive team statistics and insights
        
        ### 📈 **Interactive Analysis**
        - Select any team from the scraped data
        - View detailed tactical breakdowns
        - Export data in JSON and CSV formats
        
        ---
        
        **🔧 To get started:**
        1. Update the x-mas header in the sidebar (required for API access)
        2. Enter league configuration details
        3. Click "Start Scraping" or upload a JSON file
        4. Select a team to analyze
        
        **📝 Example League Configurations:**
        - **Premier League:** ID: `47`, Slug: `premier-league`
        - **2. Bundesliga:** ID: `146`, Slug: `2.-bundesliga`
        - **La Liga:** ID: `87`, Slug: `laliga`
        - **Serie A:** ID: `55`, Slug: `serie-a`
        """)


if __name__ == "__main__":
    main()