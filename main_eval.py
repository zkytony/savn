from __future__ import print_function, division

import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.multiprocessing as mp
import queue

import time
import numpy as np
import random
import json
from tqdm import tqdm

from utils.net_util import ScalarMeanTracker
from runners import nonadaptivea3c_val, savn_val


def main_eval(args, create_shared_model, init_agent):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    model_to_open = args.load_model

    processes = []

    res_queue = queue.Queue()
    if args.model == "BaseModel" or args.model == "GCN":
        args.learned_loss = False
        args.num_steps = 50
        target = nonadaptivea3c_val
    else:
        args.learned_loss = True
        args.num_steps = 6
        target = savn_val

    rank = 0
    max_count = 250

    func_calls = []
    for scene_type in args.scene_types:
        target_args = (rank,
                       args,
                       model_to_open,
                       create_shared_model,
                       init_agent,
                       res_queue,
                       max_count,
                       scene_type)
        func_calls.append((target, target_args))

    count = 0
    end_count = 0
    train_scalars = ScalarMeanTracker()
    pbar = tqdm(total=max_count * len(args.scene_types))
    try:
        for target, target_args in func_calls:
            target(*target_args, pbar)
            train_result = res_queue.get()
            count += 1
            if "END" in train_result:
                end_count += 1
                continue
            train_scalars.add_scalars(train_result)
        tracked_means = train_scalars.pop_and_reset()
    finally:
        with open(args.results_json, "w") as fp:
            json.dump(tracked_means, fp, sort_keys=True, indent=4)
