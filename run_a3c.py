import argparse
import copy
import multiprocessing as mp
import os
import sys
import statistics
import time
import random

import chainer
from chainer import links as L
from chainer import functions as F
import cv2
import numpy as np

import a3c
import random_seed
import async
from prepare_output_dir import prepare_output_dir


def eval_performance(process_idx, make_env, model, phi, n_runs):
    assert n_runs > 1, 'Computing stdev requires at least two runs'
    scores = []
    fs_list = []
    env = make_env(process_idx, test=True)
    for i in range(n_runs):
        env.reset()
        model.reset_state()
        obs = env.reset()
        done = False
        test_r = 0
        action_count = 0
        fss = 0
        while not done:
            s = chainer.Variable(np.expand_dims(phi(obs), 0).astype(np.float32))
            pout, _ = model.pi_and_v(s)
            fs = model.fs(s, pout.action_indices).frameskip[0]
            a = pout.action_indices[0]
            obs, r, done, info = env.step(a, frameskip=fs, eval=True)
            test_r += r
            action_count += 1
            fss += fs
        scores.append(test_r)
        fs_list.append(fss/action_count)
        print('test_{}:'.format(i), test_r, ' ave_fs{}:'.format(i), fss/action_count)
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stdev = statistics.stdev(scores)
    afs_mean = statistics.mean(fs_list)
    afs_median = statistics.median(fs_list)
    afs_stdev = statistics.stdev(fs_list)
    return mean, median, stdev, afs_mean, afs_median, afs_stdev


def train_loop(process_idx, counter, make_env, max_score, args, agent, env,
               start_time, outdir):
    try:

        total_r = 0
        episode_r = 0
        action_times = 0
        global_t = 0
        local_t = 0
        obs = env.reset()
        r = 0
        done = False

        while True:

            # Get and increment the global counter
            with counter.get_lock():
                counter.value += 1
                global_t = counter.value
            local_t += 1

            if global_t > args.steps:
                break

            agent.optimizer.lr = (
                args.steps - global_t - 1) / args.steps * args.lr

            total_r += r
            episode_r += r
            # Get action and frameskip
            a, frameskip = agent.act(obs, r, done)
            action_times += 1

            if done:
                if process_idx == 0:
                    elapsed = time.time() - start_time
                    speed = global_t / elapsed * 60 * 60 / 1000000
                    print('{} global_t:{} local_t:{} lr:{} r:{} speed:{:.2f}M/hour action_times:{}'.format(
                        outdir, global_t, local_t, agent.optimizer.lr,
                        episode_r, speed, action_times))
                episode_r = 0
                action_times = 0
                obs = env.reset()
                r = 0
                done = False
            else:
                obs, r, done, info = env.step(a, frameskip=frameskip)

            if global_t % args.eval_frequency == 0:
                # Evaluation
                # We must use a copy of the model because test runs can change
                # the hidden states of the model
                test_model = copy.deepcopy(agent.model)
                test_model.reset_state()

                mean, median, stdev, afs_mean, afs_median, afs_stdev = eval_performance(
                    process_idx, make_env, test_model, agent.phi,
                    args.eval_n_runs)
                with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
                    elapsed = time.time() - start_time
                    record = (global_t, elapsed, mean, median, stdev, afs_mean, afs_median, afs_stdev)
                    print('\t'.join(str(x) for x in record), file=f)
                with max_score.get_lock():
                    if mean > max_score.value:
                        # Save the best model so far
                        print('The best score is updated {} -> {}'.format(
                            max_score.value, mean))
                        filename = os.path.join(
                            outdir, '{}.h5'.format(global_t))
                        agent.save_model(filename)
                        print('Saved the current best model to {}'.format(
                            filename))
                        max_score.value = mean

    except KeyboardInterrupt:
        if process_idx == 0:
            # Save the current model before being killed
            agent.save_model(os.path.join(
                outdir, '{}_keyboardinterrupt.h5'.format(global_t)))
            print('Saved the current model to {}'.format(
                outdir), file=sys.stderr)
        raise

    if global_t == args.steps + 1:
        # Save the final model
        agent.save_model(
            os.path.join(outdir, '{}_finish.h5'.format(args.steps)))
        print('Saved the final model to {}'.format(outdir))


def train_loop_with_profile(process_idx, counter, make_env, max_score, args,
                            agent, env, start_time, outdir):
    import cProfile
    cmd = 'train_loop(process_idx, counter, make_env, max_score, args, ' \
        'agent, env, start_time)'
    cProfile.runctx(cmd, globals(), locals(),
                    'profile-{}.out'.format(os.getpid()))


def run_a3c(processes, make_env, model_opt, phi, t_max=1, beta=1e-2,
            profile=False, steps=8 * 10 ** 7, eval_frequency=10 ** 6,
            eval_n_runs=10, args={}):

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    outdir = prepare_output_dir(args, None)

    print('Output files are saved in {}'.format(outdir))

    # n_actions = 20 * 20

    model, opt = model_opt()

    shared_params = async.share_params_as_shared_arrays(model)
    shared_states = async.share_states_as_shared_arrays(opt)

    max_score = mp.Value('f', np.finfo(np.float32).min)
    counter = mp.Value('l', 0)
    start_time = time.time()

    # Write a header line first
    with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
        column_names = ('steps', 'elapsed', 'mean', 'median', 'stdev')
        print('\t'.join(column_names), file=f)

    def run_func(process_idx):
        env = make_env(process_idx, test=False)
        model, opt = model_opt()
        async.set_shared_params(model, shared_params)
        async.set_shared_states(opt, shared_states)

        agent = a3c.A3C(model, opt, t_max, 0.99, beta=beta,
                        process_idx=process_idx, phi=phi)

        if profile:
            train_loop_with_profile(process_idx, counter, make_env, max_score,
                                    args, agent, env, start_time,
                                    outdir=outdir)
        else:
            train_loop(process_idx, counter, make_env, max_score,
                       args, agent, env, start_time, outdir=outdir)

    async.run_async(processes, run_func)
