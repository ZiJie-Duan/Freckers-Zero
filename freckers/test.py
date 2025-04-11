
t = """
[IterManagerMultiProcess:3]: Simulation Init Finish
end within 69 steps
player: 0, Win: 1
end within 69 steps
player: 0, Win: 1
end within 69 steps
player: 0, Win: 1
end within 69 steps
player: 0, Win: 1
end within 69 steps
player: 0, Win: 1
end within 69 steps
player: 0, Win: 1
end within 69 steps
player: 0, Win: 1
end within 69 steps
player: 0, Win: 1
end within 69 steps
player: 0, Win: 1
end within 69 steps
player: 0, Win: 1
end within 69 steps
player: 0, Win: 1
end within 69 steps
player: 0, Win: 1
end within 69 steps
player: 0, Win: 1
end within 69 steps
player: 0, Win: 1
end within 69 steps
player: 0, Win: 1
end within 67 steps
player: 0, Win: 1
end within 67 steps
player: 0, Win: 1
end within 67 steps
player: 0, Win: 1
end within 67 steps
player: 0, Win: 1
end within 67 steps
player: 0, Win: 1
end within 67 steps
player: 0, Win: 1
end within 67 steps
player: 0, Win: 1
end within 67 steps
player: 0, Win: 1
end within 67 steps
player: 0, Win: 1
end within 67 steps
player: 0, Win: 1
end within 67 steps
player: 0, Win: 1
end within 67 steps
player: 0, Win: 1
end within 67 steps
player: 0, Win: 1
end within 67 steps
player: 0, Win: 1
end within 67 steps
player: 0, Win: 1
end within 116 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 116 steps
player: 1, Win: 1
end within 71 steps
player: 0, Win: 1
end within 71 steps
player: 0, Win: 1
end within 71 steps
player: 0, Win: 1
end within 71 steps
player: 0, Win: 1
end within 71 steps
player: 0, Win: 1
end within 71 steps
player: 0, Win: 1
end within 71 steps
player: 0, Win: 1
end within 71 steps
player: 0, Win: 1
end within 71 steps
player: 0, Win: 1
end within 71 steps
player: 0, Win: 1
end within 71 steps
player: 0, Win: 1
end within 71 steps
player: 0, Win: 1
end within 71 steps
player: 0, Win: 1
end within 71 steps
player: 0, Win: 1
end within 71 steps
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