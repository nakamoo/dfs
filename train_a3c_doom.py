import argparse
import multiprocessing as mp

import chainer
from chainer import links as L
from chainer import functions as F
from chainer import serializers
from chainer import cuda
import cv2
import numpy as np
from PIL import Image

import policy
import v_function
import dqn_head
import a3c
import random_seed
import rmsprop_async
from init_like_torch import init_like_torch
import run_a3c
import doom_env


class A3CFF(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_actions):
        self.head = dqn_head.NIPSDQNHead(n_input_channels=4)
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        self.fs_p = policy.GaussianPolicy(
            self.head.n_output_channels, n_actions)
        super().__init__(self.head, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        out = self.head(state)
        return self.pi(out), self.v(out)

    def fs(self, state, action_indices, keep_same_state=False):
        out = self.head(state)
        return self.fs_p(out, action_indices)


class A3CLSTM(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_actions):
        self.head = dqn_head.NIPSDQNHead(n_input_channels=3)
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        self.lstm = L.LSTM(self.head.n_output_channels,
                           self.head.n_output_channels)
        super().__init__(self.head, self.lstm, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        out = self.head(state)
        if keep_same_state:
            prev_h, prev_c = self.lstm.h, self.lstm.c
            out = self.lstm(out)
            self.lstm.h, self.lstm.c = prev_h, prev_c
        else:
            out = self.lstm(out)
        return self.pi(out), self.v(out)

    def reset_state(self):
        self.lstm.reset_state()
        self.head.reset_state()

    def unchain_backward(self):
        self.lstm.h.unchain_backward()
        self.lstm.c.unchain_backward()


def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eval-frequency', type=int, default=2 * 10 ** 4)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--use-lstm', action='store_true')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.set_defaults(use_lstm=False)
    args = parser.parse_args()

    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >= 0 else np

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    # Simultaneously launching multiple vizdoom processes makes program stuck,
    # so use the global lock
    env_lock = mp.Lock()

    def make_env(process_idx, test):
        with env_lock:
            return doom_env.Env()

    # FIXME: handling from env
    n_actions = 4

    def model_opt():
        if args.use_lstm:
            model = A3CLSTM(n_actions)
        else:
            model = A3CFF(n_actions)

        serializers.load_hdf5("trained_model/80000000_finish.h5", model)

        if args.gpu >= 0:
            model.to_gpu(args.gpu)
        opt = rmsprop_async.RMSpropAsync(lr=args.lr, eps=1e-1, alpha=0.99)
        opt.setup(model.v)
        opt.setup(model.fs_p)

        opt.add_hook(chainer.optimizer.GradientClipping(40))
        return model, opt

    run_a3c.run_a3c(args.processes, make_env, model_opt, lambda x: x, t_max=args.t_max,
                    beta=args.beta, profile=args.profile, steps=args.steps,
                    eval_frequency=args.eval_frequency,
                    eval_n_runs=args.eval_n_runs, args=args)


if __name__ == '__main__':
    main()
