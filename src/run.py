import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import copy
import json
import numpy as np

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.virtual_buffer import VirtualBuffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
        config_str = json.dumps(vars(args), indent=4)
        with open(os.path.join(tb_exp_direc, "config.json"), "w") as f:
            f.write(config_str)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger, unique_token=unique_token)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)
    
def test_generalization(args, task2runner, learner, unique_token):
    start_time = time.time()
    
    bs = args.batch_size
    train_center = {}
    test_task_lst = []
    for task, runner in task2runner.items():
        if task not in learner.mac.agent.task2head:
            test_task_lst.append(task)
            continue
        train_mu_lst = []
        for i in range(bs):
            batch = runner.run(test_mode=False, pred_task=task)
            mu, z, individual_z = runner.get_mu(batch)
            train_mu_lst.append(mu)
        train_center[task] = th.cat(train_mu_lst, dim = 0).mean(dim = 0)
        runner.close_env()

    ret_ours = []
    ret_random = []
    ret_owl = []
    won_ours = []
    won_random = []
    won_owl = []
    for test_task in test_task_lst:
        print(f"{test_task}", end = " ")
        test_runner = task2runner[test_task]
        all_agents_pred_task_lst = None
        pred_task_lst_ = [test_runner.probing(train_center) for _ in range(args.probe_nepisode)]
        n_agents = len(pred_task_lst_[0])
        all_agents_pred_task_lst = []
        for agent_id in range(n_agents):
            pred_task_lst = [x[agent_id][0] for x in pred_task_lst_]
            dist_lst = [x[agent_id][1] for x in pred_task_lst_]
            mu_lst = [x[agent_id][2] for x in pred_task_lst_]
            min_dist = min(dist_lst)
            idx = dist_lst.index(min_dist)
            pred_task = pred_task_lst[idx]
            mu = mu_lst[idx]
            all_agents_pred_task_lst.append(pred_task)
        print("Prediction:", all_agents_pred_task_lst[1:])
        for i in range(args.test_nepisode):
            res = test_runner.run(test_mode=True, pred_task=all_agents_pred_task_lst[1:], mode='ctde')
            if 'tuple' in str(type(res)):
                ret, won = res
                ret_ours += ret
                if won is not None:
                    won_ours += won
        for i in range(args.test_nepisode):
            res = test_runner.run_test_bandit()
            if 'tuple' in str(type(res)):
                ret, won = res
                ret_owl += ret
                if won is not None:
                    won_owl += won
        for i in range(args.test_nepisode):
            res = test_runner.run(test_mode=True, mode='random_head')
            if 'tuple' in str(type(res)):
                ret, won = res
                ret_random += ret
                if won is not None:
                    won_random += won
    tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
    tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
    args.ours_ret = np.array(ret_ours).mean()
    args.owl_ret = np.array(ret_owl).mean()
    args.random_ret = np.array(ret_random).mean()
    args.ours_won = np.array(won_ours).mean()
    args.owl_won = np.array(won_owl).mean()
    args.random_won = np.array(won_random).mean()
    config_str = json.dumps(vars(args), indent=4)
    with open(os.path.join(tb_exp_direc, "results.json"), "w") as f:
        f.write(config_str)
    print(time_str(time.time() - start_time))
    return

def identify(args, new_task_runner, task2head, task2center_z, task2radius):

    bs = args.batch_size
        
    seen_task_dist = {}
    for seen_task, _ in task2head.items():
        if seen_task not in task2center_z:
            continue
        mu_lst = []
        for i in range(bs):
            batch = new_task_runner.run_identify(test_mode=False, pred_task=seen_task)
            mu, _, _ = new_task_runner.get_mu(batch)
            mu_lst.append(mu)
        new_task_mu = th.cat(mu_lst, dim = 0)
        center = task2center_z[seen_task].unsqueeze(0).repeat(bs, 1)
        l2_dist = th.sqrt(th.sum((new_task_mu - center) ** 2, dim = -1))
        l2_dist = l2_dist.mean()
        seen_task_dist[seen_task] = l2_dist
    new_task_runner.close_env()
    if len(seen_task_dist) == 0:
        return None, 0
        
    closet_seen_task, smallest_dist = sorted(seen_task_dist.items(), key=lambda x:x[1])[0]
    if smallest_dist < task2radius[closet_seen_task] * 1.5:
        return closet_seen_task, smallest_dist.item()
    else:
        return None, 0

def run_sequential(args, logger, unique_token):

    train_tasks = args.train_tasks
    args.n_tasks = len(train_tasks)
    main_args = copy.deepcopy(args)
    task2args, task2runner, task2empty_buffer = {}, {}, {}
    task2scheme, task2groups, task2preprocess = {}, {}, {}
    for task in train_tasks:
        task_args = copy.deepcopy(args)
        if task_args.env == "sc2":
            task_args.env_args["map_name"] = task
        elif task_args.env == "gymma" or task_args.env == "lbf":
            task_args.env_args["key"] = task
        else:
            assert 0
        
        task2args[task] = task_args
        task_runner = r_REGISTRY[args.runner](args=task_args, logger=logger, task=task)
        task2runner[task] = task_runner
        
        # Set up schemes and groups here
        env_info = task_runner.get_env_info()
        task_args.n_agents = env_info["n_agents"]
        task_args.n_actions = env_info["n_actions"]
        task_args.state_shape = env_info["state_shape"]

        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
            "tau": {"vshape": (args.rnn_hidden_dim,), "group": "agents"},
        }
        groups = {
            "agents": task_args.n_agents
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=task_args.n_actions)])
        }

        task2empty_buffer[task] = ReplayBuffer(scheme, groups, task_args.buffer_size, env_info["episode_limit"] + 1,
                            preprocess=preprocess,
                            device="cpu" if task_args.buffer_cpu_only else task_args.device)
        
        # store task information
        task2scheme[task], task2groups[task], task2preprocess[task] = scheme, groups, preprocess

    # get buffer.scheme for each task
    task2buffer_scheme = {
        task: task2empty_buffer[task].scheme for task in train_tasks
    }
    task2seen = {
        task: False for task in train_tasks
    }

    virtual_buffer = VirtualBuffer(main_args.probing_buffer_size)

    # define mac
    mac = mac_REGISTRY[main_args.mac](train_tasks, task2buffer_scheme, task2args, main_args)

    # setup runner
    for task in train_tasks:
        # setup runner
        task2runner[task].setup(scheme=task2scheme[task], groups=task2groups[task], preprocess=task2preprocess[task], mac=mac)

    # define learner
    # learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    learner = le_REGISTRY[main_args.learner](mac, logger, main_args)

    if main_args.use_cuda:
        learner.cuda()
    
    task_id = -1
    total_t_env = 0
    last_change_task_T = -args.change_task_interval - 1

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        for task, runner in task2runner.items():
            runner.t_env = timestep_to_load

        test_generalization(args, task2runner, learner, unique_token)

        return
        
    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    
    last_update_probing_T = -args.update_probing_interval - 1
    last_save_head_T = -args.save_head_interval - 1

    start_time = time.time()
    last_time = start_time

    if args.t_max == -1:
        args.t_max = args.change_task_interval * len(train_tasks)
    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    merge_task = None

    while total_t_env <= args.t_max:
        
        if total_t_env - last_change_task_T >= args.change_task_interval:

            if task_id >= 0:

                if args.checkpoint_path == "":
                    model_save_time = total_t_env
                    save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(total_t_env))
                    os.makedirs(save_path, exist_ok=True)
                    logger.console_logger.info("Saving models to {}".format(save_path))
                    learner.save_models(save_path)

                mac.agent.save_head(task)
                if main_args.reset_head:
                    mac.agent.reset_head()
                if main_args.reg:
                    learner.reg_init(task)
            task_id = (task_id + 1) % len(train_tasks)
            last_change_task_T = total_t_env

            task = train_tasks[task_id]
            
            task2seen[task] = True
            runner = task2runner[task]
            runner.t_env = total_t_env
            buffer = copy.deepcopy(task2empty_buffer[task])
            
            original_task = task
            if args.probe_nepisode > 0:
                merge_task, dist = identify(args, task2runner[task], mac.agent.task2head, mac.task2center_z, mac.task2radius)
            if merge_task:
                mac.agent.load_head(merge_task)
                task = merge_task
                if args.checkpoint_path == "":
                    virtual_buffer.merge_task(merge_task, original_task)
                logger.log_stat("merge_dist_log", dist, total_t_env)

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        total_t_env = runner.t_env
        buffer.insert_episode_batch(episode_batch)
        if args.probe_nepisode > 0 and args.checkpoint_path == "":
            virtual_buffer.insert_episode_batch(episode_batch, task, original_task)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            is_first_task = sum(task2seen.values()) == 1
            learner.train(episode_sample, total_t_env, episode, task=task, is_first_task=is_first_task)

        if args.probe_nepisode > 0 and \
            total_t_env - last_update_probing_T >= args.update_probing_interval:
            if task not in mac.agent.task2head:
                mac.agent.save_head(task)
            if total_t_env - last_save_head_T >= args.save_head_interval:
                mac.agent.save_head(task)
                last_save_head_T = total_t_env
            if args.checkpoint_path == "" and virtual_buffer.can_sample(args.batch_size):
                probing_batch_dict = virtual_buffer.sample(args.batch_size)
                learner.train_probing(probing_batch_dict, total_t_env)
                last_update_probing_T = total_t_env

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (total_t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(total_t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, total_t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = total_t_env
            for test_task in train_tasks:
                if task2seen[test_task]:
                    test_runner = task2runner[test_task]
                    test_runner.t_env = total_t_env
                    all_agents_pred_task_lst = None
                    if args.probe_nepisode > 0 and len(mac.task2center_z) > 1:
                        pred_task_lst_ = [test_runner.probing(mac.task2center_z) for _ in range(args.probe_nepisode)]
                        n_agents = len(pred_task_lst_[0])
                        all_agents_pred_task_lst = []
                        for agent_id in range(n_agents):
                            pred_task_lst = [x[agent_id][0] for x in pred_task_lst_]
                            dist_lst = [x[agent_id][1] for x in pred_task_lst_]
                            mu_lst = [x[agent_id][2] for x in pred_task_lst_]
                            min_dist = min(dist_lst)
                            idx = dist_lst.index(min_dist)
                            pred_task = pred_task_lst[idx]
                            mu = mu_lst[idx]
                            if agent_id == 0:
                                logger.log_stat(f"oracle_pred_task{train_tasks.index(test_task)}", train_tasks.index(pred_task), total_t_env)
                            else:
                                logger.log_stat(f"agent{agent_id - 1}_pred_task{train_tasks.index(test_task)}", train_tasks.index(pred_task), total_t_env)
                            all_agents_pred_task_lst.append(pred_task)
                    for i in range(n_test_runs):
                        test_runner.run(test_mode=True, mode='oracle')
                    if all_agents_pred_task_lst:
                        for i in range(n_test_runs):
                            test_runner.run(test_mode=True, pred_task=all_agents_pred_task_lst[1:], mode='ctde')
                        for i in range(n_test_runs):
                            test_runner.run(test_mode=True, mode='random_head')

        if args.save_model and (total_t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = total_t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(total_t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (total_t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, total_t_env)

            try:
                logger.log_stat("n_heads", len(mac.agent.task2head), total_t_env)
                logger.log_stat("n_big_buffers", virtual_buffer.n_big_buffers(), total_t_env)
                logger.log_stat("n_small_buffers", virtual_buffer.n_small_buffers(), total_t_env)
                if merge_task:
                    logger.log_stat("merged_task", train_tasks.index(merge_task), total_t_env)
                else:
                    logger.log_stat("merged_task", train_tasks.index(task), total_t_env)
            except:
                pass

            logger.print_recent_stats()
            last_log_T = total_t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")

def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
