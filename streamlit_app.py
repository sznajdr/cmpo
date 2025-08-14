import streamlit as st
import json
import pandas as pd
import numpy as np
import requests
import pickle
from collections import defaultdict, Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="FotMob Team Analysis",
    page_icon="⚽",
    layout="wide"
)

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

    def load_optimized_data(self, pkl_url):
        """Load pre-processed optimized data from PKL file URL"""
        try:
            response = requests.get(pkl_url, timeout=60)
            response.raise_for_status()
            
            # Load pickle data from response content
            self.data = pickle.loads(response.content)
            
            # Extract team names (same as before)
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
            st.error(f"❌ Error loading optimized data: {e}")
            return False

    def load_data(self, json_data):
        """Load data from JSON - kept for backward compatibility"""
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

        # 3. SUBSTITUTION IMPACT ANALYSIS
        report.append(f"\n🔄 SUBSTITUTION IMPACT ANALYSIS")
        report.append("-" * 50)

        # Get players with significant substitute appearances
        sub_players = []
        for player_id, player_data in team_data['player_pool'].items():
            if player_data['sub_appearances'] >= 2:  # At least 2 sub appearances
                sub_info = {
                    'name': player_data['name'],
                    'position': player_data['primary_position'],
                    'sub_apps': player_data['sub_appearances'],
                    'avg_sub_minute': player_data['avg_sub_minute'],
                    'sub_minute_range': player_data['sub_minute_range'],
                    'goals': player_data['goals'],
                    'assists': player_data['assists'],
                    'xG': player_data['xG'],
                    'avg_rating': player_data['avg_rating'],
                    'start_rate': player_data['start_rate']
                }

                # Calculate substitute-specific metrics if available
                if player_id in team_data.get('substitution_analysis', {}):
                    sub_analysis = team_data['substitution_analysis'][player_id]
                    sub_info.update({
                        'results_when_subbed': sub_analysis['results_when_subbed'],
                        'avg_rating_as_sub': sub_analysis['avg_rating_as_sub']
                    })

                sub_players.append(sub_info)

        if sub_players:
            # Sort by sub appearances
            sub_players.sort(key=lambda x: x['sub_apps'], reverse=True)

            report.append("🌟 TOP IMPACT SUBSTITUTES:")
            for i, player in enumerate(sub_players[:10], 1):
                role_type = "🟠 Super-Sub" if player['start_rate'] < 50 else "🟡 Rotation"

                impact_stats = []
                if player['goals'] > 0 or player['assists'] > 0:
                    impact_stats.append(f"{player['goals']}G+{player['assists']}A")
                if player['avg_rating'] > 0:
                    impact_stats.append(f"{player['avg_rating']:.1f}★")

                impact_display = " | " + " | ".join(impact_stats) if impact_stats else ""

                sub_timing = f"avg {player['avg_sub_minute']:.0f}'" if player['avg_sub_minute'] > 0 else "varied timing"
                if player['sub_minute_range']:
                    sub_timing += f" (range: {player['sub_minute_range']})"

                report.append(f"  {i:2d}. {player['name']} ({player['position']}) {role_type}")
                report.append(f"      📊 {player['sub_apps']} sub apps | {sub_timing}{impact_display}")

                # Show results when subbed if available
                if 'results_when_subbed' in player:
                    results = player['results_when_subbed']
                    record = f"{results['W']}W-{results['D']}D-{results['L']}L"
                    report.append(f"      📈 Team record when subbed: {record}")
        else:
            report.append("⚠️ No regular substitute players found")

        # 4. COMPREHENSIVE TEAM STATISTICS
        report.append(f"\n📊 COMPREHENSIVE TEAM STATISTICS")
        report.append("-" * 60)

        team_matches = team_data['matches']

        if team_matches:
            # Basic stats
            total_matches = len(team_matches)
            total_goals_for = sum(match['team_score'] for match in team_matches)
            total_goals_against = sum(match['opponent_score'] for match in team_matches)

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

            # Calculate basic averages
            avg_goals_for = total_goals_for / total_matches
            avg_goals_against = total_goals_against / total_matches

            # Record
            results = [match['result'] for match in team_matches]
            wins = results.count('W')
            draws = results.count('D')
            losses = results.count('L')
            record = f"{wins}W-{draws}D-{losses}L"
            points = wins * 3 + draws

            # Display comprehensive stats
            report.append(f"🏆 Overall Record: {record} in {total_matches} matches ({points} points)")
            report.append(f"📊 Goal Statistics:")
            report.append(f"   • Goals For: {total_goals_for} ({avg_goals_for:.2f}/game)")
            report.append(f"   • Goals Against: {total_goals_against} ({avg_goals_against:.2f}/game)")
            report.append(f"   • Goal Difference: {total_goals_for - total_goals_against:+d} ({(avg_goals_for - avg_goals_against):+.2f}/game)")

            report.append(f"⚽ Shooting Statistics:")
            if team_averages.get('total_shots', 0) > 0:
                report.append(f"   • Shots For: {team_averages['total_shots']:.1f}/game | Against: {opponent_averages.get('total_shots', 0):.1f}/game")
                report.append(f"   • Shots on Target For: {team_averages.get('shots_on_target', 0):.1f}/game | Against: {opponent_averages.get('shots_on_target', 0):.1f}/game")
                report.append(f"   • Big Chances For: {team_averages.get('big_chances', 0):.1f}/game | Against: {opponent_averages.get('big_chances', 0):.1f}/game")

                # Shot accuracy
                if team_averages['total_shots'] > 0:
                    shot_accuracy = (team_averages.get('shots_on_target', 0) / team_averages['total_shots']) * 100
                    report.append(f"   • Shot Accuracy: {shot_accuracy:.1f}%")

                # Conversion rate
                if team_averages.get('shots_on_target', 0) > 0:
                    conversion_rate = (avg_goals_for / team_averages['shots_on_target']) * 100
                    report.append(f"   • Conversion Rate: {conversion_rate:.1f}% (goals/shots on target)")

            report.append(f"🎯 Possession & Passing:")
            if team_averages.get('ball_possession', 0) > 0:
                report.append(f"   • Average Possession: {team_averages['ball_possession']:.1f}%")
            if team_averages.get('accurate_passes', 0) > 0:
                report.append(f"   • Accurate Passes For: {team_averages['accurate_passes']:.0f}/game | Against: {opponent_averages.get('accurate_passes', 0):.0f}/game")

            report.append(f"⚠️ Discipline & Set Pieces:")
            if team_averages.get('fouls_committed', 0) > 0:
                report.append(f"   • Fouls For: {team_averages['fouls_committed']:.1f}/game | Against: {opponent_averages.get('fouls_committed', 0):.1f}/game")
            if team_averages.get('corners', 0) > 0:
                report.append(f"   • Corners For: {team_averages['corners']:.1f}/game | Against: {opponent_averages.get('corners', 0):.1f}/game")

            # Home vs Away split
            home_matches = [m for m in team_matches if m['is_home']]
            away_matches = [m for m in team_matches if not m['is_home']]

            if home_matches and away_matches:
                home_goals = sum(m['team_score'] for m in home_matches) / len(home_matches)
                away_goals = sum(m['team_score'] for m in away_matches) / len(away_matches)
                home_results = [m['result'] for m in home_matches]
                away_results = [m['result'] for m in away_matches]
                home_record = f"{home_results.count('W')}W-{home_results.count('D')}D-{home_results.count('L')}L"
                away_record = f"{away_results.count('W')}W-{away_results.count('D')}D-{away_results.count('L')}L"

                report.append(f"🏠 Home vs Away Performance:")
                report.append(f"   • Home: {home_record} ({home_goals:.2f} goals/game)")
                report.append(f"   • Away: {away_record} ({away_goals:.2f} goals/game)")

            # Formation preferences
            formation_counts = {}
            for match in team_matches:
                formation = match['formation']
                if formation:
                    formation_counts[formation] = formation_counts.get(formation, 0) + 1

            if formation_counts:
                report.append(f"🏟️ Formation Usage:")
                for formation, count in sorted(formation_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_matches) * 100
                    formation_data = team_data['formations'].get(formation, {})
                    win_rate = formation_data.get('win_rate', 0)
                    report.append(f"   • {formation}: {count} times ({percentage:.1f}%) - {win_rate:.1f}% win rate")

        # 5. TACTICAL INSIGHTS
        report.append(f"\n🧠 TACTICAL INSIGHTS")
        report.append("-" * 40)

        # Best formation
        best_formation = None
        best_win_rate = 0
        for formation, data in team_data['formations'].items():
            if data['usage_count'] >= 3 and data['win_rate'] > best_win_rate:
                best_win_rate = data['win_rate']
                best_formation = formation

        if best_formation:
            report.append(f"🏆 Most Successful Formation: {best_formation} ({best_win_rate:.1f}% win rate)")

        # Key players identification
        key_starters = [p for p in team_data['player_pool'].values() if p['start_rate'] > 70 and p['starts'] >= 5]
        if key_starters:
            top_performer = max(key_starters, key=lambda x: x['avg_rating'])
            report.append(f"⭐ Top Performer: {top_performer['name']} ({top_performer['avg_rating']:.1f} avg rating)")

        # Super sub identification
        super_subs = [p for p in team_data['player_pool'].values() if p['sub_appearances'] >= 3 and p['start_rate'] < 50]
        if super_subs:
            best_sub = max(super_subs, key=lambda x: x['xG'])
            report.append(f"🟠 Best Super-Sub: {best_sub['name']} ({best_sub['xG']:.1f} xG)")

        # Most used formation
        most_used = max(team_data['formations'].items(), key=lambda x: x[1]['usage_count'])
        report.append(f"📊 Preferred Formation: {most_used[0]} ({most_used[1]['usage_count']} times)")

        report.append(f"\n{'='*80}")
        
        return "\n".join(report)


def main():
    st.title("⚽ FotMob Team Analysis - Optimized PKL Version")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedTeamTacticalPredictor()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    # Data loading section
    st.subheader("📊 Data Loading")
    
    # Create columns for different loading options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("🚀 **Lightning Fast PKL Loading**: This app uses pre-processed data for instant analysis!")
        
        if st.button("⚡ Load Optimized PKL Data", type="primary", use_container_width=True):
            try:
                with st.spinner("Loading optimized PKL data..."):
                    pkl_url = "https://github.com/sznajdr/cmpo/raw/refs/heads/main/pkljson.pkl"
                    
                    if st.session_state.analyzer.load_optimized_data(pkl_url):
                        st.session_state.data_loaded = True
                        st.success(f"✅ Loaded optimized data for {len(st.session_state.analyzer.team_names)} teams")
                        st.balloons()
                    else:
                        st.error("❌ Could not load PKL data")
                        st.session_state.data_loaded = False
                        
            except Exception as e:
                st.error(f"❌ Failed to load PKL data: {str(e)}")
                st.session_state.data_loaded = False
    
    with col2:
        st.write("**Alternative Data Sources:**")
        
        # Fallback sample data options
        with st.expander("🌐 Sample Data (Fallback)"):
            sample_options = {
                "POL1": "https://raw.githubusercontent.com/sznajdr/cmpo/refs/heads/main/POL1.json",
                "DE2": "https://raw.githubusercontent.com/sznajdr/cmpo/refs/heads/main/DE2.json", 
                "DE3": "https://raw.githubusercontent.com/sznajdr/cmpo/refs/heads/main/trimmed_DE3.json"
            }
            
            for name, url in sample_options.items():
                if st.button(f"📊 {name}", use_container_width=True):
                    try:
                        with st.spinner(f"Loading {name} sample data..."):
                            response = requests.get(url, timeout=30)
                            response.raise_for_status()
                            json_data = response.json()
                            
                            if st.session_state.analyzer.load_data(json_data):
                                st.session_state.data_loaded = True
                                st.success(f"✅ Loaded {name} data for {len(st.session_state.analyzer.team_names)} teams")
                            else:
                                st.error(f"❌ Could not process {name} sample data")
                                st.session_state.data_loaded = False
                                
                    except Exception as e:
                        st.error(f"❌ Failed to load {name} data: {str(e)}")
                        st.session_state.data_loaded = False
        
        # File upload
        with st.expander("📤 Upload File"):
            uploaded_file = st.file_uploader("Upload FotMob JSON file", type="json")
            
            if uploaded_file is not None:
                try:
                    with st.spinner("Loading data from file..."):
                        json_data = json.load(uploaded_file)
                        
                        if st.session_state.analyzer.load_data(json_data):
                            st.session_state.data_loaded = True
                            st.success(f"✅ Loaded data for {len(st.session_state.analyzer.team_names)} teams")
                        else:
                            st.error("❌ Could not process the uploaded file")
                            st.session_state.data_loaded = False
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    st.session_state.data_loaded = False

    # Team analysis section
    if st.session_state.data_loaded and st.session_state.analyzer.team_names:
        st.divider()
        st.subheader("🔍 Team Analysis")
        
        # Display data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Teams", len(st.session_state.analyzer.team_names))
        with col2:
            st.metric("🏟️ Total Matches", len(st.session_state.analyzer.data))
        with col3:
            st.metric("⚡ Load Speed", "Instant!")
        
        # Team selection
        selected_team = st.selectbox(
            "Choose a team to analyze:",
            options=st.session_state.analyzer.team_names,
            key="team_selector"
        )
        
        if selected_team:
            # Analysis button
            col1, col2 = st.columns([1, 3])
            
            with col1:
                analyze_button = st.button(
                    f"🔍 Analyze {selected_team}", 
                    type="primary", 
                    use_container_width=True
                )
            
            with col2:
                st.info("💡 **Tip**: Analysis is now lightning fast thanks to optimized PKL data!")
            
            if analyze_button:
                with st.spinner(f"Analyzing {selected_team}..."):
                    report = st.session_state.analyzer.create_team_report(selected_team)
                
                # Display the report in a code block to preserve formatting
                st.subheader(f"📋 {selected_team} - Tactical Analysis Report")
                st.code(report, language=None)
                
                # Download options
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name=f"{selected_team.replace(' ', '_')}_tactical_analysis.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    # Create a simplified CSV version for Excel
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
                            label="📊 Download CSV",
                            data=csv_string,
                            file_name=f"{selected_team.replace(' ', '_')}_player_data.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                with col3:
                    st.success("✅ **Analysis Complete!** Report generated instantly using optimized data.")
    
    elif not st.session_state.data_loaded:
        st.info("👆 **Get Started**: Click 'Load Optimized PKL Data' above for instant analysis!")
        
        # Show benefits of PKL loading
        st.subheader("🚀 Why Use Optimized PKL Data?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **⚡ Lightning Fast**
            - Instant loading
            - No JSON parsing
            - Optimized structure
            """)
        
        with col2:
            st.info("""
            **🎯 Clean Data**
            - Pre-processed 
            - No waste data
            - Ready for analysis
            """)
        
        with col3:
            st.info("""
            **🔧 Same Features**
            - All original functions
            - Same detailed reports
            - Better performance
            """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚽ Enhanced Team Tactical Predictor - Optimized PKL Version</p>
        <p>🚀 Lightning fast analysis with pre-processed data</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
