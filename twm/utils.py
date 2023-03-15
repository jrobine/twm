import collections
import math

import gym
import numpy as np
import torch
from torch import nn, optim
from ale_py.roms.utils import rom_name_to_id
from ale_py.env.gym import AtariEnv
from wandb.sdk.data_types.base_types.wb_value import WBValue


def update_metrics(metrics, new_metrics, prefix=None):
    def process(key, t):
        if isinstance(t, (int, float)):
            return t
        assert torch.is_tensor(t), key
        assert not t.requires_grad, key
        assert t.ndim == 0 or t.shape == (1,), key
        return t.clone()

    if prefix is None:
        metrics.update({key: process(key, value) for key, value in new_metrics.items()})
    else:
        metrics.update({f'{prefix}{key}': process(key, value) for key, value in new_metrics.items()})
    return metrics


def combine_metrics(metrics, prefix=None):
    result = {}
    if prefix is None:
        for met in metrics:
            update_metrics(result, met)
    else:
        for met, pre in zip(metrics, prefix):
            update_metrics(result, met, pre)
    return result


def mean_metrics(metrics_history, except_keys=None):
    if len(metrics_history) == 0:
        return {}
    if len(metrics_history) == 1:
        return metrics_history[0]
    except_keys = set() if except_keys is None else set(except_keys)
    result = {}
    value_history = collections.defaultdict(lambda: [])
    for metrics in metrics_history:
        for key, value in metrics.items():
            if key in except_keys or isinstance(value, WBValue):
                result[key] = value  # use last value
            else:
                value_history[key].append(value)
    result.update({key: compute_mean(values) for key, values in value_history.items()})
    return result


class MetricsSummarizer:

    def __init__(self, except_keys=None):
        self.metrics_history = []
        self.except_keys = set() if except_keys is None else set(except_keys)

    def append(self, metrics):
        self.metrics_history.append(metrics)

    def summarize(self):
        summary = mean_metrics(self.metrics_history, except_keys=self.except_keys)
        self.metrics_history = []
        return summary


def compute_mean(values):
    if torch.is_tensor(values):
        return values.float().mean()
    if isinstance(values, (tuple, list)):
        return torch.stack([torch.as_tensor(x).detach() for x in values]).float().mean()
    raise ValueError()


def random_choice(n, num_samples, replacement=False, device=None):
    if replacement:
        return torch.randint(0, n, (num_samples,), device=device)

    weights = torch.ones(n, device=device)
    return torch.multinomial(weights, num_samples, replacement=False)


def windows(x, window_size, window_stride=1):
    x = x.unfold(1, window_size, window_stride)
    dims = list(range(x.ndim))[:-1]
    dims.insert(2, x.ndim - 1)
    x = x.permute(dims)
    return x


def same_batch_shape(tensors, ndim=2):
    batch_shape = tensors[0].shape[:ndim]
    assert all(t.ndim >= ndim for t in tensors)
    return all(tensors[i].shape[:ndim] == batch_shape for i in range(1, len(tensors)))


def same_batch_shape_time_offset(a, b, offset):
    assert a.ndim >= 2 and b.ndim >= 2
    return a.shape[:2] == (b.shape[0], b.shape[1] + offset)


def check_no_grad(*tensors):
    return all((t is None or not t.requires_grad) for t in tensors)


class AdamOptim:

    def __init__(self, parameters, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, grad_clip=0):
        self.parameters = list(parameters)
        self.grad_clip = grad_clip
        self.optimizer = optim.Adam(self.parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.parameters, self.grad_clip)
        self.optimizer.step()


def create_reward_transform(transform_type):
    if transform_type == 'tanh':
        def transform(r):
            if torch.is_tensor(r):
                return torch.tanh(r)
            return math.tanh(r)
    elif transform_type == 'clip':
        def transform(r):
            if torch.is_tensor(r):
                return torch.clip(r, -1, 1)
            return np.clip(r, -1, 1)
    elif transform_type == 'none' or transform_type is None:
        def transform(r):
            return r
    else:
        raise ValueError(transform_type)
    return transform


def preprocess_atari_obs(obs, device=None):
    if isinstance(obs, gym.wrappers.LazyFrames):
        obs = np.array(obs)
    return torch.as_tensor(obs, device=device).float() / 255.


def create_atari_env(game, noop_max=30, frame_skip=4, frame_stack=4, frame_size=84,
                     episodic_lives=True, grayscale=True, time_limit=27000):
    env = AtariEnv(rom_name_to_id(game), frameskip=1, repeat_action_probability=0.0)
    env.spec = gym.spec(game + 'NoFrameskip-v4')  # required for AtariPreprocessing
    has_fire_action = env.get_action_meanings()[1] == 'FIRE'
    env = gym.wrappers.AtariPreprocessing(
        env, noop_max=0 if has_fire_action else noop_max, frame_skip=frame_skip, screen_size=frame_size,
        terminal_on_life_loss=False, grayscale_obs=grayscale)
    if has_fire_action:
        env = FireAfterLifeLoss(env)
        if noop_max > 0:
            env = NoopStart(env, noop_max)  # noops after fire
    if episodic_lives:
        # return done when a life is lost, but don't reset environment until no lives are left
        env = EpisodicLives(env)
    env = gym.wrappers.FrameStack(env, frame_stack)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=time_limit)
    return env


def create_vector_env(num_envs, env_fn):
    if num_envs == 1:
        return gym.vector.SyncVectorEnv([env_fn])
    else:
        return gym.vector.AsyncVectorEnv([env_fn for _ in range(num_envs)])


# from https://github.com/mila-iqia/spr/blob/release/src/rlpyt_utils.py
atari_human_scores = {
    'Alien': 7127.7, 'Amidar': 1719.5, 'Assault': 742.0, 'Asterix': 8503.3, 'BankHeist': 753.1, 'BattleZone': 37187.5,
    'Boxing': 12.1, 'Breakout': 30.5, 'ChopperCommand': 7387.8, 'CrazyClimber': 35829.4, 'DemonAttack': 1971.0,
    'Freeway': 29.6, 'Frostbite': 4334.7, 'Gopher': 2412.5, 'Hero': 30826.4, 'Jamesbond': 302.8, 'Kangaroo': 3035.0,
    'Krull': 2665.5, 'KungFuMaster': 22736.3, 'MsPacman': 6951.6, 'Pong': 14.6, 'PrivateEye': 69571.3, 'Qbert': 13455.0,
    'RoadRunner': 7845.0, 'Seaquest': 42054.7, 'UpNDown': 11693.2
}
atari_random_scores = {
    'Alien': 227.8, 'Amidar': 5.8, 'Assault': 222.4, 'Asterix': 210.0, 'BankHeist': 14.2, 'BattleZone': 2360.0,
    'Boxing': 0.1, 'Breakout': 1.7, 'ChopperCommand': 811.0, 'CrazyClimber': 10780.5, 'DemonAttack': 152.1,
    'Freeway': 0.0, 'Frostbite': 65.2, 'Gopher': 257.6, 'Hero': 1027.0, 'Jamesbond': 29.0, 'Kangaroo': 52.0,
    'Krull': 1598.0, 'KungFuMaster': 258.5, 'MsPacman': 307.3, 'Pong': -20.7, 'PrivateEye': 24.9, 'Qbert': 163.9,
    'RoadRunner': 11.5, 'Seaquest': 68.4, 'UpNDown': 533.4
}


def compute_atari_hns(game, agent_score):
    random_score = atari_random_scores[game]
    human_score = atari_human_scores[game]
    return (agent_score - random_score) / (human_score - random_score) * 100.0


class EpisodicLives(gym.Wrapper):
    # https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

    def __init__(self, env):
        super().__init__(env)
        self.ale = env.unwrapped.ale
        self.lives = 0
        self.was_real_done = True

    def reset(self, seed=None, options=None):
        if self.was_real_done or (options is not None and options.get('force', False)):
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            obs, _, _, _, info = self.env.step(0)  # noop
        self.lives = self.ale.lives()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.ale.lives()
        if lives < self.lives and lives > 0:  # special case for Qbert
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info


class NoAutoReset(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.final_observation = None
        self.final_info = None

    def reset(self, seed=None, options=None):
        if self.final_observation is None or (options is not None and options.get('force', False)):
            return self.env.reset(seed=seed, options=options)
        return self.final_observation, self.final_info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.final_observation = obs
            self.final_info = info
        return obs, reward, terminated, truncated, info


class FireAfterLifeLoss(gym.Wrapper):
    # https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

    def __init__(self, env):
        super().__init__(env)
        unwrapped = env.unwrapped
        action_meanings = unwrapped.get_action_meanings()
        assert action_meanings[1] == 'FIRE'
        assert len(action_meanings) >= 3
        self.ale = unwrapped.ale
        self.lives = 0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(seed=seed, options=options)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(seed=seed, options=options)

        self.lives = self.ale.lives()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = self.ale.lives()
        if lives < self.lives:
            obs, reward2, terminated2, truncated2, info2 = self.env.step(1)
            reward += reward2
            terminated = terminated or terminated2
            truncated = truncated or truncated2
            info.update(info2)
        self.lives = lives
        return obs, reward, terminated, truncated, info


class NoopStart(gym.Wrapper):

    def __init__(self, env, noop_max):
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self, seed=None, options=None):
        # taken from gym.wrappers.AtariPreprocessing
        obs, reset_info = self.env.reset(seed=seed, options=options)
        noops = self.env.unwrapped.np_random.integers(1, self.noop_max + 1) if self.noop_max > 0 else 0
        for _ in range(noops):
            obs, _, terminated, truncated, step_info = self.env.step(0)
            reset_info.update(step_info)
            if terminated or truncated:
                obs, reset_info = self.env.reset(seed=seed, options=options)
        return obs, reset_info


@torch.no_grad()
def make_grid(tensor, nrow, padding, pad_value=0):
    # modified version of torchvision.utils.make_grid that supports different paddings for x and y
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding[0]), int(tensor.size(3) + padding[1])
    num_channels = tensor.size(1)
    grid = tensor.new_full(
        (num_channels, height * ymaps + padding[0], width * xmaps + padding[1]), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding[0], height - padding[0]) \
                .narrow(2, x * width + padding[1], width - padding[1]) \
                .copy_(tensor[k])
            k = k + 1
    return grid


def to_image(tensor):
    from PIL import Image
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
    if tensor.shape[2] == 1:
        tensor = tensor.squeeze(2)
    return Image.fromarray(tensor.numpy()).convert('RGB')
