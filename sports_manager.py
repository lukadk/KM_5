import os
import pandas as pd
from rdflib import Graph, Namespace, RDF
import subprocess
import json
import argparse

class SquadAnalyzer:
    def __init__(self, kg_path='knowledge_graph.ttl', stats_path='players.csv', output_file='analysis_output.txt'):
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        kg_full_path = os.path.join(script_dir, kg_path)
        stats_full_path = os.path.join(script_dir, stats_path)
        
        # Initialize knowledge graph
        self.graph = Graph()
        self.graph.parse(kg_full_path, format='turtle')
        self.ex = Namespace("http://example.org/football#")
        
        # Load player stats
        self.stats_df = pd.read_csv(stats_full_path)
        
        # Store the output file name
        self.output_file = output_file

    def get_user_inputs(self):
        """Get user inputs from the console"""
        formations = ['4-3-3', '4-4-2', '3-5-2', '3-4-3', '5-3-2']
        priorities = ['Attack', 'Defense', 'Balance']

        print("Select Formation:")
        for idx, formation in enumerate(formations, 1):
            print(f"{idx}. {formation}")
        formation_choice = int(input("Enter the number of your choice: "))
        self.formation = formations[formation_choice - 1]

        print("\nSelect Priority:")
        for idx, priority in enumerate(priorities, 1):
            print(f"{idx}. {priority}")
        priority_choice = int(input("Enter the number of your choice: "))
        self.priority = priorities[priority_choice - 1]

        # Proceed to analysis
        self.analyze_squad()

    def analyze_squad(self):
        """Main analysis function"""
        # Get data from knowledge graph
        kg_data = self.get_kg_data()
        
        # Get LLM analysis
        llm_data = self.get_llm_analysis(kg_data)
        
        # Display combined analysis
        self.display_hybrid_analysis(kg_data, llm_data)

    def get_kg_data(self):
        """Get relevant data from knowledge graph"""
        # Query available players
        player_query = """
        PREFIX ex: <http://example.org/football#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?name ?positionName ?injured ?redCard ?yellowCards
        WHERE {
            ?player rdf:type ex:Player ;
                    ex:hasName ?name ;
                    ex:canPlayPosition ?position ;
                    ex:injured ?injured ;
                    ex:redCard ?redCard ;
                    ex:yellowCards ?yellowCards .
            BIND(REPLACE(STR(?position), "^.*[#/]", "") AS ?positionName)
        }
        """
        
        players = []
        for row in self.graph.query(player_query):
            player_stats = self.stats_df[self.stats_df['name'] == str(row.name)]
            if not player_stats.empty:
                players.append({
                    'name': str(row.name),
                    'position': str(row.positionName),
                    'rating': float(player_stats.iloc[0]['rating']) if pd.notna(player_stats.iloc[0]['rating']) else None,
                    'status': {
                        'injured': str(row.injured).lower() == 'true',
                        'redCard': str(row.redCard).lower() == 'true',
                        'yellowCards': int(row.yellowCards)
                    }
                })

        return {
            'available_players': players,
            'formation': self.formation,
            'priority': self.priority
        }

    def get_llm_analysis(self, kg_data):
        """Get tactical analysis from LLM"""
        prompt = f"""
Analyze this football team data and provide specific tactical recommendations.

Formation: {kg_data['formation']}
Priority: {kg_data['priority']}

Available Players:
{self._format_players_for_prompt(kg_data['available_players'])}

Return a JSON with:
{{
    "best_lineup": {{
        "keeper": ["1-2 names"],
        "defenders": ["3-5 names based on formation"],
        "midfielders": ["3-5 names based on formation"],
        "forwards": ["2-3 names based on formation"]
    }},
    "tactical_analysis": ["3-4 specific points"],
    "key_strengths": ["2-3 points"],
    "potential_risks": ["2-3 points"]
}}

Return ONLY the JSON.
"""

        try:
            result = subprocess.run(
                ['ollama', 'run', 'llama3.2', prompt],
                capture_output=True,
                text=True,
                timeout=60  # Increase timeout if necessary
            )
            output_text = result.stdout.strip()
            start_idx = output_text.find('{')
            end_idx = output_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = output_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("Invalid JSON format")
                
        except Exception as e:
            print(f"Note: Using rule-based analysis as LLM failed: {str(e)}")
            return self._get_fallback_analysis(kg_data)

    def _get_fallback_analysis(self, kg_data):
        """Generate rule-based analysis when LLM fails"""
        # Sort players by rating
        sorted_players = {}
        for player in kg_data['available_players']:
            if not player['status']['injured'] and not player['status']['redCard']:
                pos = player['position']
                if pos not in sorted_players:
                    sorted_players[pos] = []
                sorted_players[pos].append(player)
        
        # Sort each position by rating
        for pos in sorted_players:
            sorted_players[pos].sort(
                key=lambda x: (x['rating'] if x['rating'] is not None else 0),
                reverse=True
            )

        # Select best players based on formation
        formation = kg_data['formation']
        requirements = {
            '4-3-3': {'Keeper': 1, 'Defender': 4, 'Midfielder': 3, 'Forward': 3},
            '4-4-2': {'Keeper': 1, 'Defender': 4, 'Midfielder': 4, 'Forward': 2},
            '3-5-2': {'Keeper': 1, 'Defender': 3, 'Midfielder': 5, 'Forward': 2},
            '3-4-3': {'Keeper': 1, 'Defender': 3, 'Midfielder': 4, 'Forward': 3},
            '5-3-2': {'Keeper': 1, 'Defender': 5, 'Midfielder': 3, 'Forward': 2}
        }

        req = requirements[formation]
        best_lineup = {
            'keeper': [p['name'] for p in sorted_players.get('Keeper', [])[:req['Keeper']]],
            'defenders': [p['name'] for p in sorted_players.get('Defender', [])[:req['Defender']]],
            'midfielders': [p['name'] for p in sorted_players.get('Midfielder', [])[:req['Midfielder']]],
            'forwards': [p['name'] for p in sorted_players.get('Forward', [])[:req['Forward']]]
        }

        return {
            'best_lineup': best_lineup,
            'tactical_analysis': [
                f"Using {formation} formation",
                f"Focus on {kg_data['priority'].lower()} play",
                "Build from solid defensive base",
                "Utilize available player strengths"
            ],
            'key_strengths': [
                "Best players selected for each position",
                "Formation matches available players",
                "Balanced team structure"
            ],
            'potential_risks': [
                "Players adapting to formation",
                "Tactical coordination needed",
                "Fitness management required"
            ]
        }

    def display_hybrid_analysis(self, kg_data, analysis):
        """Display combined analysis results in the console and save to a file"""
        output_lines = []
        output_lines.append(f"{'='*50}")
        output_lines.append(f"{kg_data['formation']} Formation Analysis ({kg_data['priority']})")
        output_lines.append(f"{'-'*50}")
        output_lines.append("Recommended Lineup:")
        for position, players in analysis['best_lineup'].items():
            player_list = ', '.join(players) if players else 'No available players'
            output_lines.append(f"{position.capitalize()}: {player_list}")
        output_lines.append(f"\nTactical Analysis:")
        for point in analysis['tactical_analysis']:
            output_lines.append(f"- {point}")
        output_lines.append(f"\nKey Strengths:")
        for strength in analysis['key_strengths']:
            output_lines.append(f"- {strength}")
        output_lines.append(f"\nPotential Risks:")
        for risk in analysis['potential_risks']:
            output_lines.append(f"- {risk}")
        output_lines.append(f"{'='*50}\n")
        
        # Print to console
        print('\n'.join(output_lines))
        
        # Save to file
        with open(self.output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"Analysis saved to {self.output_file}")

    def _get_player_rating(self, player_name):
        """Get player rating from stats DataFrame"""
        player_stats = self.stats_df[self.stats_df['name'] == player_name]
        if not player_stats.empty and pd.notna(player_stats.iloc[0]['rating']):
            return float(player_stats.iloc[0]['rating'])
        return None

    def _format_players_for_prompt(self, players):
        """Format player data for LLM prompt"""
        return "\n".join([
            f"{p['name']}: {p['position']}, Rating: {p['rating']:.2f}" if p['rating'] else f"{p['name']}: {p['position']}, Rating: N/A"
            for p in players
            if not (p['status']['injured'] or p['status']['redCard'])
        ])

def main():
    parser = argparse.ArgumentParser(description='Squad Analyzer')
    parser.add_argument('--kg', type=str, default='knowledge_graph.ttl', help='Path to knowledge graph file')
    parser.add_argument('--stats', type=str, default='players.csv', help='Path to player stats CSV file')
    parser.add_argument('--output', type=str, default='analysis_output.txt', help='Output file name')
    args = parser.parse_args()
    
    analyzer = SquadAnalyzer(kg_path=args.kg, stats_path=args.stats, output_file=args.output)
    analyzer.get_user_inputs()

if __name__ == "__main__":
    main()

