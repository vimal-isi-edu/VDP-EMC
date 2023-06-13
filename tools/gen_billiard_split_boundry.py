"""
This script comes from the RTRBM code by Ilya Sutskever from
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
"""
import os
import pickle
import hickle
import scipy.io
import numpy as np
from numpy import *
from scipy import *
from tqdm import tqdm
import torch

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

shape_std = shape
# size_h = 64
# size_w = 64
size_h = 96
size_w = 192
size = [size_h, size_w]
base_v = 2.0
# friction = 1.0
T = 100
visible_size = T
num_objs = 3
r = 2.0
G = 9.81
FPS = 30

with_action = True
# action policy: [random action, srctar action]
random_action = False
sample_action = True
sample_act_theta = np.arange(0, 12) / 12.0 * np.pi * 2
# < v8
# sample_act_vel = np.array([0.25, 0.5, 1.0, 2.0, 4.0]) * base_v
sample_act_vel = np.array([2.0, 3.0, 4.0, 5.0, 6.0])



def shape(A):
    if isinstance(A, ndarray):
        return shape_std(A)
    else:
        return A.shape()

def new_speeds(m1, m2, v1, v2):
    new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2


def norm(x): return sqrt((x ** 2).sum())


def bounce_n(T=128, n=2, r=None, m=None, ball_move_size=None, adjust_split_line=None):
    if m is None:
        m = array([1] * n)
    # r is to be rather small.
    X = zeros((T, n, 2), dtype='float')
    y = zeros((T, n, 2), dtype='float')
    init_a = zeros((n, 3), dtype='float')
    good_config = False

    while not good_config:
        x = rand(n, 2) * ball_move_size
        good_config = True
        for i in range(n):
            for z in range(2):
                if x[i][z] - r[i] < 0:
                    good_config = False
                if x[i][z] + r[i] > ball_move_size[z] - 1:
                    good_config = False
            if x[i][1] - r[i] < adjust_split_line[0] and x[i][1] + r[i] >= adjust_split_line[0]:
                good_config = False
            if x[i][1] - r[i] <= adjust_split_line[1] and x[i][1] + r[i] > adjust_split_line[1]:
                good_config = False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(x[i] - x[j]) < r[i] + r[j]:
                    good_config = False

    if with_action:
        v = 0 * randn(n, 2)
        src = np.random.randint(n)
        if random_action:
            v_val = randn(2)
            v_val = v_val / norm(v_val)
            v[src, :] = v_val * base_v
            init_a[src, 0] = v_val[0]
            init_a[src, 1] = v_val[1]
            init_a[src, 2] = base_v
        elif sample_action:
            src = np.random.randint(n)
            act_theta = np.random.choice(sample_act_theta)
            act_vel = np.random.choice(sample_act_vel)
            v[src, :] = np.sin(act_theta), np.cos(act_theta)
            v = v / (norm(v) + 1e-5) * act_vel
            init_a[src, 0] = np.sin(act_theta)
            init_a[src, 1] = np.cos(act_theta)
            init_a[src, 2] = act_vel
        else:
            src = np.random.randint(n)
            tar = np.random.randint(n)
            while tar == src:
                tar = np.random.randint(n)
            v[src, :] = x[tar] - x[src]
            v = v / (norm(v) + 1e-5) * base_v
            init_a = v
    else:
        v = randn(n, 2)
        v = v / norm(v) * base_v

    eps = .2
    for t in range(T):
        # for how long do we show small simulation

        v_prev = copy(v)

        for i in range(n):
            X[t, i] = x[i]
            y[t, i] = v[i]

        col_judge = np.diag([1, 1, 1])
        for mu in range(int(1 / eps)):

            for i in range(n):
                x[i] += eps * v[i]

            for i in range(n):
                for z in range(2):
                    if x[i][z] - r[i] < 0:
                        v[i][z] = abs(v[i][z])  # want positive
                    if x[i][z] + r[i] > ball_move_size[z] - 1:
                        v[i][z] = -abs(v[i][z])  # want negative
                if x[i][1] - r[i] < adjust_split_line[0] and x[i][1] + r[i] > adjust_split_line[0]:
                    v[i][1] = -abs(v[i][1])
                if x[i][1] + r[i] > adjust_split_line[1] and x[i][1] - r[i] < adjust_split_line[1]:
                    v[i][1] = abs(v[i][1])

            for i in range(n):
                for j in range(i):
                    if norm(x[i] - x[j]) < r[i] + r[j] and col_judge[i, j] == 0:
                        # the bouncing off part:
                        w = x[i] - x[j]
                        w = w / norm(w)

                        v_i = dot(w.transpose(), v[i])
                        v_j = dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                        v[i] += w * (new_v_i - v_i)
                        v[j] += w * (new_v_j - v_j)

                        col_judge[i, j] = 1
                        col_judge[j, i] = 1


    return X, y, init_a


def ar(x, y, z):
    return z / 2 + arange(x, y, z, dtype='float')


def draw_image(X, res, r=None, board_size=None, split_line=None):
    T, n = shape(X)[0:2]
    A = zeros((visible_size, res[0], res[1], 3), dtype='float')
    # B = ones((visible_size, res[0], res[1], 3), dtype='float') * 0.5

    [X_perm, Y_perm] = meshgrid(ar(0, 1, 1. / res[1]) * size[1], ar(0, 1, 1. / res[0]) * size[0])
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    board_color = [0.5, 0.5, 0.5]
    for t in range(visible_size):
        for i in range(n):
            A[t, :, :, 0] += colors[i][0] * exp(-(((Y_perm - X[t, i, 0]) ** 2 +
                                                   (X_perm - X[t, i, 1]) ** 2) /
                                                  (r[i] ** 2)) ** 4)
            A[t, :, :, 1] += colors[i][1] * exp(-(((Y_perm - X[t, i, 0]) ** 2 +
                                                   (X_perm - X[t, i, 1]) ** 2) /
                                                  (r[i] ** 2)) ** 4)
            A[t, :, :, 2] += colors[i][2] * exp(-(((Y_perm - X[t, i, 0]) ** 2 +
                                                   (X_perm - X[t, i, 1]) ** 2) /
                                                  (r[i] ** 2)) ** 4)
        A[t][A[t] > 1] = 1
        # A[t][A[t] < 100/255] = 0
        # A[t, bg_mask[0], bg_mask[1], :] = 0.5
    # bg_mask = A == [0,0,0]
    # A[bg_mask] = B[bg_mask]
    A[:, 0:board_size[0], :] = board_color
    if board_size[2] != 0:
        A[:, -board_size[2]:, :] = board_color
    A[:, :, 0:board_size[1]] = board_color
    if board_size[3] != 0:
        A[:, :, -board_size[3]:] = board_color
    A[:, :, split_line[0]:split_line[1]+1] = board_color

    B = zeros((visible_size, res[0], res[1], 2), dtype='float')
    B[:, :, :] = [0, 1]
    board_mask = [1, 0]
    B[:, 0:board_size[0], :] = board_mask
    if board_size[2] != 0:
        B[:, -board_size[2]:, :] = board_mask
    B[:, :, 0:board_size[1]] = board_mask
    if board_size[3] != 0:
        B[:, :, -board_size[3]:] = board_mask
    B[:, :, split_line[0]:split_line[1]+1] = board_mask
    # A = np.where(A[:,:,:], [0, 0, 0], [0.5, 0.5, 0.5])
    return A, B


def bounce_vec(res, n=2, T=128, r=None, m=None, board_size=None, ball_move_size=None, split_line=None):
    valid = False
    while not valid:
        # x, y, init_a, key_frame, col_frame = bounce_n(T, n, r, m)
        adjust_split_line = split_line - board_size[1]
        x, y, init_a = bounce_n(T, n, r, m, ball_move_size, adjust_split_line)
        x[:, :, 1] += board_size[1]
        x[:, :, 0] += board_size[0]
        valid = True
        # for test planning, task 1: only one ball moving for the first 50 timesteps
        # pre_valid = (np.absolute(y)[:40].sum((0, 2)) > 1e-3).sum() < 3
        # valid = (np.absolute(y)[40:60].sum((0, 2)) > 1e-3).sum() == 3
        # valid = pre_valid and valid
        V, board_mask = draw_image(x, res, r, board_size, split_line)
        y = concatenate((x, y), axis=2)
        # return V.reshape(visible_size, res[0], res[1], 3), y, init_a, key_frame, col_frame
        return V.reshape(visible_size, res[0], res[1], 3), y, init_a, board_mask.reshape(visible_size, res[0], res[1], 2)
# make sure you have this folder


def show_sample(V, i):
    logdir = f'./debug/img_{i}'
    os.makedirs(logdir, exist_ok=True)
    T = V.shape[0]
    for t in range(T):
        plt.imshow(V[t])
        # Save it
        fname = logdir + '/' + str(t) + '.png'
        plt.savefig(fname)


def gen_data(N, num_objs, name, show_video=False):
    res = size

    dat = empty((N, visible_size, res[0], res[1], 3), dtype=float)
    dat_mask = empty((N, visible_size, res[0], res[1], 2), dtype=float)
    dat_y = empty((N, T, num_objs, 4), dtype=float)
    init_a = empty((N, num_objs, 3), dtype=float)
    key_f = empty((N, T, num_objs), dtype=bool)
    col_f = empty((N, T,), dtype=bool)
    dat_r = empty((N, 3))
    data_board = empty((N, 4), dtype=float)
    data_split = empty((N, 1), dtype=float)
    for i in tqdm(range(N)):
        board_size = np.random.randint(low=0, high=16, size=4)
        # board_size = np.random.randint(low=5, high=6, size=4)
        board_total_hw = [board_size[0]+board_size[2], board_size[1]+board_size[3]]
        ball_move_size = [size[0] - board_total_hw[0], size[1] - board_total_hw[1]]
        split_point = np.random.randint(low=63, high=128)
        split_size = 2
        split_line = np.array([split_point-split_size, split_point+split_size])
        r1 = 2
        r2 = 2
        r3 = 2
        r = array([r1, r2, r3])
        # dat[i], dat_y[i], init_a[i], key_f[i], col_f[i] = bounce_vec(res=res, n=num_objs, T=T, r=r)
        dat[i], dat_y[i], init_a[i], dat_mask[i] = bounce_vec(res=res, n=num_objs, T=T, r=r, board_size=board_size, ball_move_size=ball_move_size, split_line=split_line)
        dat_r[i] = r.copy()
        data_board[i] = board_size.copy()
        data_split[i] = split_point
    data = dict()
    data['X'] = dat
    data['y'] = dat_y
    data['a'] = init_a
    data['keyf'] = key_f
    data['colf'] = col_f
    data['r'] = dat_r
    data['board_mask'] = dat_mask
    data['board_size'] = data_board
    data['split_point'] = data_split


    num_key_frames = len(np.where(key_f.sum(2) > 0)[0])
    num_col_frames = len(np.where(col_f)[0])
    num_env_frames = num_key_frames - num_col_frames

    print(f'{name}: env interaction: {num_env_frames / N:.2f}, ball interaction: {num_col_frames / N:.2f}')
    with open(name, 'wb') as f:
        # pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        hickle.dump(data, f)

    # show one video
    if show_video:
        for i in range(N):
            show_sample(dat[i], i)


if __name__ == '__main__':
    cache_name = 'data/simb_split'
    os.makedirs(cache_name, exist_ok=True)
    datasets = ['train', 'test']
    samples = [1000, 1000]
    for (dataset, sample) in zip(datasets, samples):
        rng_seed = sum([ord(c) for c in dataset])
        print(rng_seed)
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        # gen_data(sample, num_objs, name=f'{cache_name}/{dataset}.pkl', show_video=False)
        gen_data(sample, num_objs, name=f'{cache_name}/{dataset}.hkl', show_video=False)
