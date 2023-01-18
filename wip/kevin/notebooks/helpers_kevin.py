from itertools import product
from pathlib import Path
import glob
import os


import numpy as np
import pandas as pd


DATA = "/Users/caldera/Documents/EPFL/GAP/Project/Projet-Gameplay/wip/kevin/data"

# right, left, jump, fire
REMAP = {"Key.right": "R", "Key.left": "L", "'d'": "J", "Key.space": "J", "Key.up": "J"}

FPS = 60
N_FRAME = [2 * FPS, 10 * FPS, 60 * FPS, 300 * FPS]
BLOCK_TIME = ["2s", "10s", "1min", "5min"]
PATTERN_MAX_LENGTH = [3, 4, 10, 15]
PATTERNS = ["".join(p) for n in range(2, 15) for p in product("RLJ", repeat=n)]


def slice_keylog_file(frames_and_inputs_list, predicted=True):

    for i, run in enumerate(frames_and_inputs_list):
        for block_time, block_size in zip(BLOCK_TIME, N_FRAME):
            block = list()
            n_block = int(run.FRAME.max() / block_size)

            for period in range(n_block):
                slice = run[
                    (run.FRAME >= period * block_size) & (run.FRAME < (period + 1) * block_size)
                ]
                block.append(slice)

                if predicted:
                    save_path = DATA + "/sliced_logs/predicted/run_{}/{}/slice{}.csv".format(
                        i, block_time, period
                    )
                else:
                    save_path = DATA + "/sliced_logs/ground_truth/run_{}/{}/slice{}.csv".format(
                        i, block_time, period
                    )

                filepath2 = Path(save_path)
                filepath2.parent.mkdir(parents=True, exist_ok=True)
                slice.to_csv(filepath2)


# read csv file (inputs), return list of pressed keys (down only and standardized as LRJ)
# slices the file au passage
def read_and_standardize_logs(csv_file, predicted=True):
    if predicted:
        save_path = DATA + "/keylogs/predicted"
    else:
        save_path = DATA + "/keylogs/ground_truth"

    frames_and_inputs_list = list()

    data = pd.read_csv(csv_file)
    # data['run_number'] = i

    main_keys = (
        data.KEY.value_counts().keys()[:4].tolist()
    )  # might not be necessary for predicted inputs
    data = data[data.KEY.isin(main_keys)]

    keylogs = data[
        (data.KEY != "'a'") & (data.KEY != "Key.enter") & (data.KEY != "'s'")
    ]  # A modifier quand on saura ce que predit exactement le modele
    keylogs = keylogs.replace({"KEY": REMAP})  # idem
    keylogs = keylogs[keylogs.STATUS == "DOWN"]  # on garde que les touches pressées atm

    frames_and_inputs_list.append(keylogs)

    slice_keylog_file(frames_and_inputs_list, predicted)

    return frames_and_inputs_list


# inputs_log : must contain a "KEY" column with only letter L,R,J
# return Series of pattern count in decreasing order; the pattern is the index
def get_pattern_count(standardized_inputs, max_pattern_size, min_pattern_occur=10):
    d = dict()
    keys_as_str = standardized_inputs.KEY.str.cat

    for pattern in [element for element in PATTERNS if len(element) <= max_pattern_size]:
        d[pattern] = keys_as_str.count(pattern)

    frequencies = pd.Series(dict(sorted(d.items(), key=lambda x: x[1], reverse=True)))

    return frequencies[frequencies > min_pattern_occur]
