
t = """
end within 120 steps
player: 1, Win: 1
end within 123 steps
player: 0, Win: 1
end within 130 steps
player: 1, Win: 1
end within 152 steps
player: 1, Win: 1
end within 126 steps
player: 1, Win: 1
end within 125 steps
player: 0, Win: 1
end within 132 steps
player: 1, Win: 1
end within 104 steps
player: 1, Win: 1
end within 107 steps
player: 0, Win: 1
end within 121 steps
player: 0, Win: 1
end within 130 steps
player: 1, Win: 1
end within 137 steps
player: 0, Win: 1
end within 131 steps
player: 0, Win: 1
end within 126 steps
player: 1, Win: 1
end within 126 steps
player: 1, Win: 1
end within 150 steps
player: 1, Win: 1
end within 128 steps
player: 1, Win: 1
end within 126 steps
player: 1, Win: 1
end within 130 steps
player: 1, Win: 1
end within 137 steps
player: 0, Win: 1
end within 118 steps
player: 1, Win: 1
end within 115 steps
player: 0, Win: 1
end within 134 steps
player: 1, Win: 1
end within 118 steps
player: 1, Win: 1
end within 132 steps
player: 1, Win: 1
end within 135 steps
player: 0, Win: 1
end within 114 steps
player: 1, Win: 1
end within 159 steps
player: 0, Win: 1
end within 131 steps
player: 0, Win: 1
end within 133 steps
player: 0, Win: 1
end within 113 steps
player: 0, Win: 1
end within 141 steps
player: 0, Win: 1
end within 156 steps
player: 1, Win: 1
end within 132 steps
player: 1, Win: 1
end within 140 steps
player: 1, Win: 1
end within 138 steps
player: 1, Win: 1
end within 112 steps
player: 1, Win: 1
end within 135 steps
player: 0, Win: 1
end within 136 steps
player: 1, Win: 1
end within 133 steps
player: 0, Win: 1
end within 133 steps
player: 0, Win: 1
end within 137 steps
player: 0, Win: 1
end within 121 steps
player: 0, Win: 1
end within 147 steps
player: 0, Win: 1
end within 129 steps
player: 0, Win: 1
end within 131 steps
player: 0, Win: 1
end within 154 steps
player: 1, Win: 1
end within 137 steps
player: 0, Win: 1
end within 138 steps
player: 1, Win: 1
end within 165 steps
player: 0, Win: 1
end within 145 steps
player: 0, Win: 1
end within 127 steps
player: 0, Win: 1
end within 127 steps
player: 0, Win: 1
end within 139 steps
player: 0, Win: 1
end within 126 steps
player: 1, Win: 1
end within 109 steps
player: 0, Win: 1
end within 111 steps
player: 0, Win: 1
end within 132 steps
player: 1, Win: 1
end within 108 steps
player: 1, Win: 1
end within 134 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 120 steps
player: 1, Win: 1
end within 133 steps
player: 0, Win: 1
end within 117 steps
player: 0, Win: 1
end within 133 steps
player: 0, Win: 1
end within 146 steps
player: 1, Win: 1
end within 148 steps
player: 1, Win: 1
end within 120 steps
player: 1, Win: 1
end within 124 steps
player: 1, Win: 1
end within 149 steps
player: 0, Win: 1
end within 137 steps
player: 0, Win: 1
end within 130 steps
player: 1, Win: 1
end within 126 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 150 steps
player: 1, Win: 1
end within 143 steps
player: 0, Win: 1
end within 140 steps
player: 1, Win: 1
end within 153 steps
player: 0, Win: 1
end within 131 steps
player: 0, Win: 1
end within 129 steps
player: 0, Win: 1
end within 126 steps
player: 1, Win: 1
end within 146 steps
player: 1, Win: 1
end within 148 steps
player: 1, Win: 1
end within 130 steps
player: 1, Win: 1
end within 137 steps
player: 0, Win: 1
end within 139 steps
player: 0, Win: 1
end within 129 steps
player: 0, Win: 1
end within 134 steps
player: 1, Win: 1
end within 106 steps
player: 1, Win: 1
end within 128 steps
player: 1, Win: 1
end within 115 steps
player: 0, Win: 1


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