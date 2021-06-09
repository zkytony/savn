from __future__ import division

import time
import setproctitle
import copy
from datasets.glove import Glove
from datasets.data import get_data, name_to_num

from models.model_io import ModelOptions

from .train_util import (
    new_episode,
    run_episode,
    reset_player,
    compute_spl,
    get_bucketed_metrics,
    SGD_step,
    end_episode,
    compute_loss,
    get_params,
    compute_learned_loss,
    print_debug
)

def savn_val(
    rank,
    args,
    model_to_open,
    model_create_fn,
    initialize_agent,
    res_queue,
    max_count,
    scene_type,
    pbar=None
):

    glove = Glove(args.glove_file)
    scenes, possible_targets, targets = get_data(args.scene_types, args.val_scenes)
    num = name_to_num(scene_type)
    scenes = scenes[num]
    targets = targets[num]

    if scene_type == "living_room":
        args.max_episode_length = 200
    else:
        args.max_episode_length = 100

    setproctitle.setproctitle("Training Agent: {}".format(rank))

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]

    import torch

    torch.cuda.set_device(gpu_id)

    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    shared_model = model_create_fn(args)

    if model_to_open is not None:
        saved_state = torch.load(
            model_to_open, map_location=lambda storage, loc: storage
        )
        shared_model.load_state_dict(saved_state)

    player = initialize_agent(model_create_fn, args, rank, gpu_id=gpu_id)
    player.sync_with_shared(shared_model)
    count = 0

    player = initialize_agent(model_create_fn, args, rank, gpu_id=gpu_id)

    model_options = ModelOptions()

    while count < max_count:

        count += 1
        print_debug("[%s]------start (count %d)----------------" % (scene_type, count))

        start_time = time.time()
        new_episode(args, player, scenes, possible_targets, targets, glove=glove)
        player_start_state = copy.deepcopy(player.environment.controller.state)
        player.episode.exploring = True
        total_reward = 0
        player.eps_len = 0

        # theta <- shared_initialization
        params_list = [get_params(shared_model, gpu_id)]
        model_options.params = params_list[-1]
        loss_dict = {}
        reward_dict = {}
        episode_num = 0
        num_gradients = 0

        print_debug("[%s]------running (count %d)----------------" % (scene_type, count))

        while True:
            print_debug("[%s]------running (No.Grad %d) (count %d)----------------" % (scene_type, num_gradients, count))
            args.scene_type = scene_type
            args.num_gradients = num_gradients
            args.count = count
            total_reward = run_episode(player, args, total_reward, model_options, False)

            if player.done:
                break

            if args.gradient_limit < 0 or episode_num < args.gradient_limit:

                num_gradients += 1

                # Compute the loss.
                learned_loss = compute_learned_loss(args, player, gpu_id, model_options)

                if args.verbose:
                    print("inner gradient")
                inner_gradient = torch.autograd.grad(
                    learned_loss["learned_loss"],
                    [v for _, v in params_list[episode_num].items()],
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )

                params_list.append(
                    SGD_step(params_list[episode_num], inner_gradient, args.inner_lr)
                )
                model_options.params = params_list[-1]

                # reset_player(player)
                episode_num += 1

                for k, v in learned_loss.items():
                    loss_dict["{}/{:d}".format(k, episode_num)] = v.item()

        loss = compute_loss(args, player, gpu_id, model_options)

        for k, v in loss.items():
            loss_dict[k] = v.item()
        reward_dict["total_reward"] = total_reward

        spl, best_path_length = compute_spl(player, player_start_state)
        bucketed_spl = get_bucketed_metrics(spl, best_path_length, player.success)

        print_debug("[%s]------putting (count %d)----------------" % (scene_type, count))
        end_episode(
            player,
            res_queue,
            total_time=time.time() - start_time,
            spl=spl,
            **reward_dict,
            **bucketed_spl,
        )
        if pbar is not None:
            pbar.update(1)
        print_debug("[%s]------done putting (count %d)----------------" % (scene_type, count))

        reset_player(player)

    print_debug("************************* I AM HERE! " + scene_type)

    player.exit()
    res_queue.put({"END": True})
