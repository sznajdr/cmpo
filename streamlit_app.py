import streamlit as st
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="FotMob Team Analysis",
    page_icon="⚽",
    layout="wide"
)

class FotMobAnalyzer:
    def __init__(self):
        self.data = None
        self.team_names = []

    def load_data(self, json_data):
        """Load and process JSON data"""
        try:
            self.data = json_data if isinstance(json_data, list) else [json_data]
            
            # Extract team names
            teams = set()
            for match in self.data:
                try:
                    home = match.get('general', {}).get('homeTeam', {}).get('name')
                    away = match.get('general', {}).get('awayTeam', {}).get('name')
                    if home: teams.add(home)
                    if away: teams.add(away)
                except:
                    continue
            
            self.team_names = sorted(list(teams))
            return len(teams) > 0
        except:
            return False

    def get_safe(self, obj, path, default=None):
        """Safely get nested values"""
        try:
            keys = path.split('.')
            current = obj
            for key in keys:
                current = current[key]
            return current
        except:
            return default

    def analyze_team(self, team_name):
        """Analyze specific team"""
        if not self.data:
            return None

        matches = []
        formations = defaultdict(int)
        players = defaultdict(lambda: {'starts': 0, 'subs': 0, 'name': '', 'position': ''})
        
        for match in self.data:
            try:
                home_team = self.get_safe(match, 'general.homeTeam.name')
                away_team = self.get_safe(match, 'general.awayTeam.name')
                
                if team_name not in [home_team, away_team]:
                    continue
                
                is_home = team_name == home_team
                opponent = away_team if is_home else home_team
                
                # Get scores
                teams = self.get_safe(match, 'header.teams', [])
                if len(teams) < 2:
                    continue
                
                team_score = teams[0].get('score', 0) if is_home else teams[1].get('score', 0)
                opp_score = teams[1].get('score', 0) if is_home else teams[0].get('score', 0)
                
                result = 'W' if team_score > opp_score else 'D' if team_score == opp_score else 'L'
                
                # Get lineup
                lineup_key = 'homeTeam' if is_home else 'awayTeam'
                lineup = self.get_safe(match, f'content.lineup.{lineup_key}', {})
                formation = lineup.get('formation', 'Unknown')
                
                formations[formation] += 1
                
                # Process starters
                for player in lineup.get('starters', []):
                    pid = str(player.get('id', ''))
                    players[pid]['starts'] += 1
                    players[pid]['name'] = player.get('name', '')
                    players[pid]['position'] = self.get_safe(player, 'positionLabel.label', 'Unknown')
                
                # Process subs
                for player in lineup.get('subs', []):
                    pid = str(player.get('id', ''))
                    players[pid]['subs'] += 1
                    players[pid]['name'] = player.get('name', '')
                    players[pid]['position'] = self.get_safe(player, 'positionLabel.label', 'Unknown')
                
                matches.append({
                    'opponent': opponent,
                    'result': result,
                    'score': f"{team_score}-{opp_score}",
                    'formation': formation,
                    'is_home': is_home
                })
                
            except Exception as e:
                continue
        
        if not matches:
            return None
        
        return {
            'matches': matches,
            'formations': dict(formations),
            'players': dict(players)
        }

def main():
    st.title("⚽ FotMob Team Analysis")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = FotMobAnalyzer()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    # File upload
    uploaded_file = st.file_uploader("Upload FotMob JSON file", type="json")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Loading data..."):
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

    # Team analysis
    if st.session_state.data_loaded and st.session_state.analyzer.team_names:
        st.subheader("Select Team")
        
        selected_team = st.selectbox(
            "Choose a team:",
            options=st.session_state.analyzer.team_names
        )
        
        if selected_team:
            with st.spinner(f"Analyzing {selected_team}..."):
                team_data = st.session_state.analyzer.analyze_team(selected_team)
            
            if team_data:
                # Basic stats
                matches = team_data['matches']
                total_matches = len(matches)
                wins = sum(1 for m in matches if m['result'] == 'W')
                draws = sum(1 for m in matches if m['result'] == 'D')
                losses = sum(1 for m in matches if m['result'] == 'L')
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Matches", total_matches)
                with col2:
                    st.metric("Wins", wins)
                with col3:
                    st.metric("Draws", draws)
                with col4:
                    st.metric("Losses", losses)
                
                # Tabs for different analyses
                tab1, tab2, tab3 = st.tabs(["📊 Formations", "👥 Squad", "🏆 Matches"])
                
                with tab1:
                    st.subheader("Formation Usage")
                    formations = team_data['formations']
                    
                    if formations:
                        form_df = pd.DataFrame([
                            {'Formation': form, 'Times Used': count, 'Usage %': f"{count/total_matches*100:.1f}%"}
                            for form, count in sorted(formations.items(), key=lambda x: x[1], reverse=True)
                        ])
                        st.dataframe(form_df, use_container_width=True, hide_index=True)
                        
                        # Formation chart
                        fig_data = pd.DataFrame(list(formations.items()), columns=['Formation', 'Count'])
                        st.bar_chart(fig_data.set_index('Formation'))
                    else:
                        st.info("No formation data available")
                
                with tab2:
                    st.subheader("Squad Analysis")
                    players = team_data['players']
                    
                    if players:
                        player_list = []
                        for pid, data in players.items():
                            if data['name']:
                                total_apps = data['starts'] + data['subs']
                                start_rate = (data['starts'] / total_matches * 100) if total_matches > 0 else 0
                                
                                role = "🔵 Key Player" if start_rate > 80 else \
                                       "🟡 Regular" if start_rate > 50 else \
                                       "🟠 Rotation" if start_rate > 20 else "⚪ Fringe"
                                
                                player_list.append({
                                    'Player': data['name'],
                                    'Position': data['position'],
                                    'Starts': data['starts'],
                                    'Sub Apps': data['subs'],
                                    'Total Apps': total_apps,
                                    'Start Rate %': f"{start_rate:.1f}%",
                                    'Role': role
                                })
                        
                        if player_list:
                            player_df = pd.DataFrame(player_list)
                            player_df = player_df.sort_values('Start Rate %', ascending=False)
                            st.dataframe(player_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No player data available")
                    else:
                        st.info("No squad data available")
                
                with tab3:
                    st.subheader("Match Results")
                    
                    if matches:
                        match_df = pd.DataFrame([
                            {
                                'Opponent': m['opponent'],
                                'Result': m['result'],
                                'Score': m['score'],
                                'Formation': m['formation'],
                                'Venue': '🏠 Home' if m['is_home'] else '✈️ Away'
                            }
                            for m in matches
                        ])
                        st.dataframe(match_df, use_container_width=True, hide_index=True)
                        
                        # Results summary
                        home_matches = [m for m in matches if m['is_home']]
                        away_matches = [m for m in matches if not m['is_home']]
                        
                        if home_matches and away_matches:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**🏠 Home Record**")
                                home_w = sum(1 for m in home_matches if m['result'] == 'W')
                                home_d = sum(1 for m in home_matches if m['result'] == 'D')
                                home_l = sum(1 for m in home_matches if m['result'] == 'L')
                                st.write(f"{home_w}W - {home_d}D - {home_l}L")
                            
                            with col2:
                                st.write("**✈️ Away Record**")
                                away_w = sum(1 for m in away_matches if m['result'] == 'W')
                                away_d = sum(1 for m in away_matches if m['result'] == 'D')
                                away_l = sum(1 for m in away_matches if m['result'] == 'L')
                                st.write(f"{away_w}W - {away_d}D - {away_l}L")
                    else:
                        st.info("No match data available")
                
                # Export option
                st.subheader("💾 Export Data")
                if st.button("📊 Generate CSV Report"):
                    # Create comprehensive report
                    report_data = {
                        'Team': selected_team,
                        'Total_Matches': total_matches,
                        'Wins': wins,
                        'Draws': draws,
                        'Losses': losses,
                        'Win_Rate': f"{wins/total_matches*100:.1f}%" if total_matches > 0 else "0%"
                    }
                    
                    report_df = pd.DataFrame([report_data])
                    csv = report_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Report",
                        data=csv,
                        file_name=f"{selected_team.replace(' ', '_')}_report.csv",
                        mime="text/csv"
                    )
            else:
                st.error(f"No data found for {selected_team}")
    
    elif not st.session_state.data_loaded:
        st.info("👆 Upload a FotMob JSON file to get started")

if __name__ == "__main__":
    main()
