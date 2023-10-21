import sys
from suika_player import SuikaGameController

repeat = int(sys.argv[1]) if len(sys.argv) > 1 else None

print(f"Running random game {repeat if repeat else 'infinite'} times")

suikagame = SuikaGameController(mouse_speed=10)
suikagame.run_random_game_loop(num_retries=repeat)
