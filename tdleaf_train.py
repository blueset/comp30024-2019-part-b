import os
import subprocess
import random
import logging
import pathlib
import pickle
from datetime import datetime

logging.basicConfig(format='%(message)s')

train_log = logging.getLogger("train")
train_log.addHandler(logging.FileHandler(
    "training.log"
))

weights_log = logging.getLogger("weightn")
weights_log.addHandler(logging.FileHandler(
    "weights.log"
))

log = train_log.critical
weights_log = weights_log.critical

weights_file = pathlib.Path("ᐖ⵿/weights.pkl")

PLAYERS = {
    "maxn": "ᐖ⵿",
    "greedy": "ᐖ⵿/greedy_player.py",
    "random":  "ᐖ⵿/random_player.py",
    "mcts": "ᐖ⵿/mcts_library.py"
}
iter = 10

while True:
    log("==============================")
    log(f"Iteration {iter}, {datetime.now().strftime('%c')}")
    with weights_file.open("rb") as f:
        wght = pickle.load(f)
        log(f"Starting weights: {wght}")
    players = ["maxn", "maxn", "maxn"]
    maxn_depth = {
        "cut_off_depth_red": '2',
        "cut_off_depth_green": '2', 
        "cut_off_depth_blue": '2', 
    }
    log(f"Agents: {players}")
    log(f"MaxN config: {maxn_depth}")
    players = list(map(PLAYERS.get, players))
    round = 0
    failed_count = 0
    while True:
        weight = None
        failed_reason = None
        env = os.environ.copy()
        env.update(maxn_depth)
        with subprocess.Popen(["/usr/bin/python3", "-m", "referee"] + players,
                              stdout=subprocess.PIPE, bufsize=1, 
                              universal_newlines=True,
                              env=env) as p:
            for line in p.stdout:
                if line.startswith("NEW_WEIGHT"):
                    weight = eval(line.split(maxsplit=1)[1])
                elif line.startswith("TRAIN PLAYER") or line.startswith("draw detected"):
                    log(failed_reason)
                    failed_reason = line
                    
        if weight:
            with weights_file.open("wb") as f:
                pickle.dump(weight, f)
            log(f"Round {round}: {weight}")
        else:
            log(f'Round {round}: Stopped. ({failed_reason})')
            failed_count += 1

        if failed_count >= 3:
            log('CONVERGING, break.')
            with weights_file.open("rb") as f:
                wght = pickle.load(f)
                log(f"Final weight: {wght}")
                weights_log("==============================")
                weights_log(f"Iteration {iter}: {datetime.now().strftime('%c')}")
                weights_log(f"Weights: {wght}")
                weights_log(f"Ended due to: {failed_reason}")
            break
        round += 1
 
    new_weights = [random.randint(-50, 50) for _ in range(10)]
    with weights_file.open("wb") as f:
        pickle.dump(new_weights, f)
    iter += 1

            



    
