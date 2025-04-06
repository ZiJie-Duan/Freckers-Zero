
t = """
end within 108 steps
player: 1, Win: 1
end within 112 steps
player: 1, Win: 1
end within 115 steps
player: 0, Win: 1
end within 122 steps
player: 1, Win: 1
end within 86 steps
player: 1, Win: 1
end within 100 steps
player: 1, Win: 1
end within 82 steps
player: 1, Win: 1
end within 105 steps
player: 0, Win: 1
end within 82 steps
player: 1, Win: 1
end within 102 steps
player: 1, Win: 1
end within 114 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 103 steps
player: 0, Win: 1
end within 117 steps
player: 0, Win: 1
end within 112 steps
player: 1, Win: 1
end within 118 steps
player: 1, Win: 1
end within 110 steps
player: 1, Win: 1
end within 96 steps
player: 1, Win: 1
end within 117 steps
player: 0, Win: 1
end within 118 steps
player: 1, Win: 1
end within 102 steps
player: 1, Win: 1
end within 118 steps
player: 1, Win: 1
end within 88 steps
player: 1, Win: 1
end within 112 steps
player: 1, Win: 1
end within 104 steps
player: 1, Win: 1
end within 102 steps
player: 1, Win: 1
end within 140 steps
player: 1, Win: 1
end within 108 steps
player: 1, Win: 1
end within 108 steps
player: 1, Win: 1
end within 108 steps
player: 1, Win: 1
end within 98 steps
player: 1, Win: 1
end within 126 steps
player: 1, Win: 1
end within 96 steps
player: 1, Win: 1
end within 84 steps
player: 1, Win: 1
end within 102 steps
player: 1, Win: 1
end within 138 steps
player: 1, Win: 1
end within 78 steps
player: 1, Win: 1
end within 112 steps
player: 1, Win: 1
end within 92 steps
player: 1, Win: 1
end within 124 steps
player: 1, Win: 1
end within 102 steps
player: 1, Win: 1
end within 125 steps
player: 0, Win: 1
end within 100 steps
player: 1, Win: 1
end within 106 steps
player: 1, Win: 1
end within 90 steps
player: 1, Win: 1
end within 110 steps
player: 1, Win: 1
end within 114 steps
player: 1, Win: 1
"""
# Initialize counters
player0_wins = 0
player1_wins = 0

# Split the data into lines and process each line
lines = t.strip().split('\n')
for line in lines:
    if line.startswith('player:'):
        parts = line.split(',')
        player_part = parts[0].strip()
        win_part = parts[1].strip()
        
        # Extract player number
        player = int(player_part.split(':')[1].strip())
        
        # Check if it's a win
        if win_part == 'Win: 1':
            if player == 0:
                player0_wins += 1
            elif player == 1:
                player1_wins += 1

print(f"Player 0 wins: {player0_wins}")
print(f"Player 1 wins: {player1_wins}")