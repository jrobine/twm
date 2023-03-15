import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D

import utils


class ReplayBuffer:

    def __init__(self, config, env):
        self.config = config
        self.env = env

        device = config['buffer_device']
        self.device = torch.device(device)
        self.prev_seed = (config['seed'] + 1) * 7979
        initial_obs, _ = env.reset(seed=self.prev_seed)
        initial_obs = torch.as_tensor(np.array(initial_obs), device=device)
        capacity = config['buffer_capacity']

        self.obs = torch.zeros((capacity + 1,) + initial_obs.shape, dtype=initial_obs.dtype, device=device)
        self.obs[0] = initial_obs
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float, device=device)
        self.terminated = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.truncated = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.timesteps = torch.zeros(capacity + 1, dtype=torch.long, device=device)
        self.timesteps[0] = 0
        self.sample_visits = torch.zeros(capacity, dtype=torch.long, device='cpu')  # we sample indices on cpu

        self.capacity = capacity
        self.size = 0
        self.total_reward = 0
        self.score = 0
        self.episode_lengths = []
        self.scores = []
        self.reward_transform = utils.create_reward_transform(config['env_reward_transform'])
        self.metrics_num_episodes = 0

    def sample_random_action(self):
        return self.env.action_space.sample()

    def _get(self, array, idx, device=None, prefix=0, return_next=False, repeat_fill_value=None, allow_last=False):
        assert prefix >= 0
        squeeze_seq = False
        squeeze_batch = False
        if isinstance(idx, int):
            idx = torch.tensor([idx])
            squeeze_seq = True
        if isinstance(idx, range):
            idx = tuple(idx)
        if isinstance(idx, (tuple, list, np.ndarray)):
            idx = torch.as_tensor(idx, device=self.device)

        assert torch.is_tensor(idx)
        assert torch.all(idx >= 0)
        assert torch.all((idx <= self.size) if allow_last else (idx < self.size))

        if idx.ndim == 1:
            idx = idx.unsqueeze(0)
            squeeze_batch = True

        if prefix > 0 or return_next:
            idx_list = [idx]
            if prefix > 0:
                prefix_idx = idx[:, 0].unsqueeze(-1) + torch.arange(-prefix, 0, device=idx.device)
                prefix_mask = prefix_idx < 0
                # repeat first value, if prefix goes beyond the first value in the buffer
                prefix_idx[prefix_mask] = 0
                idx_list.insert(0, prefix_idx)

            if return_next:
                last_idx = idx[:, -1]
                suffix_idx = last_idx + 1
                # repeat value, if next goes beyond the last value in the buffer
                suffix_mask = (suffix_idx > self.size) if allow_last else (suffix_idx >= self.size)
                suffix_idx = suffix_idx * (~suffix_mask) + last_idx * suffix_mask
                idx_list.append(suffix_idx.unsqueeze(1))

            idx = torch.cat(idx_list, dim=1)
            x = array[idx]

            if repeat_fill_value is not None:
                if prefix > 0:
                    tmp = x[:, :prefix]
                    tmp[prefix_mask] = repeat_fill_value
                    x[:, :prefix] = tmp
                if return_next:
                    x[suffix_mask, -1] = repeat_fill_value
        else:
            x = array[idx]

        if squeeze_seq:
            x = x.squeeze(1)
        if squeeze_batch:
            x = x.squeeze(0)
        if device is not None and x.device != device:
            return x.to(device=device)
        return x

    def get_obs(self, idx, device=None, prefix=0, return_next=False):
        obs = self._get(self.obs, idx, device, prefix, return_next=return_next, allow_last=True)
        return utils.preprocess_atari_obs(obs, device)

    def get_actions(self, idx, device=None, prefix=0):
        return self._get(self.actions, idx, device, prefix, repeat_fill_value=0)  # noop

    def get_rewards(self, idx, device=None, prefix=0):
        return self._get(self.rewards, idx, device, prefix, repeat_fill_value=0.)

    def get_terminated(self, idx, device=None, prefix=0):
        return self._get(self.terminated, idx, device, prefix)

    def get_truncated(self, idx, device=None, prefix=0):
        return self._get(self.truncated, idx, device, prefix)

    def get_timesteps(self, idx, device=None, prefix=None):
        return self._get(self.timesteps, idx, device, prefix, allow_last=True)

    def get_data(self, idx, device=None, prefix=None, return_next_obs=False):
        obs = self.get_obs(idx, device, prefix, return_next_obs)
        actions = self.get_actions(idx, device, prefix)
        rewards = self.get_rewards(idx, device, prefix)
        terminated = self.get_terminated(idx, device, prefix)
        truncated = self.get_truncated(idx, device, prefix)
        timesteps = self.get_timesteps(idx, device, prefix)
        return obs, actions, rewards, terminated, truncated, timesteps

    def step(self, policy_fn):
        config = self.config
        index = self.size
        if index >= config['buffer_capacity']:
            raise ValueError('Buffer overflow')

        action = policy_fn(index)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            # throws away last obs
            seed = self.prev_seed
            if seed is not None:
                seed = seed * 3 + 13
                self.prev_seed = seed
            next_obs, _ = self.env.reset(seed=seed)

        self.obs[index + 1] = torch.as_tensor(np.array(next_obs), device=self.device)
        self.rewards[index] = self.reward_transform(reward)
        self.actions[index] = action
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.timesteps[index + 1] = 0 if (terminated or truncated) else (self.timesteps[index] + 1)

        self.size = index + 1
        self.total_reward += reward
        self.score += reward
        if terminated or truncated:
            self.episode_lengths.append(self.timesteps[index] + 1)
            self.scores.append(self.score)
            self.score = 0

    def _compute_visit_probs(self, n):
        temperature = self.config['buffer_temperature']
        if temperature == 'inf':
            visits = self.sample_visits[:n].float()
            visit_sum = visits.sum()
            if visit_sum == 0:
                probs = torch.full_like(visits, 1 / n)
            else:
                probs = 1 - visits / visit_sum
        else:
            logits = self.sample_visits[:n].float() / -temperature
            probs = F.softmax(logits, dim=0)
        assert probs.device.type == 'cpu'
        return probs

    def sample_indices(self, max_batch_size, sequence_length):
        n = self.size - sequence_length + 1
        batch_size = max_batch_size
        if batch_size * sequence_length > n:
            raise ValueError('Not enough data in buffer')

        probs = self._compute_visit_probs(n)
        start_idx = torch.multinomial(probs, batch_size, replacement=False)

        # stay on cpu
        flat_idx = start_idx.reshape(-1)
        flat_idx, counts = torch.unique(flat_idx, return_counts=True)
        self.sample_visits[flat_idx] += counts

        start_idx = start_idx.to(device=self.device)
        idx = start_idx.unsqueeze(-1) + torch.arange(sequence_length, device=self.device)
        return idx

    def generate_uniform_indices(self, batch_size, sequence_length, extra=0):
        start_offset = random.randint(0, sequence_length - 1)
        start_idx = torch.arange(start_offset, self.size - sequence_length, sequence_length,
                                 dtype=torch.long, device=self.device)
        start_idx = start_idx[torch.randperm(start_idx.shape[0], device=self.device)]
        while len(start_idx) > 0:
            idx = start_idx[:batch_size]
            idx = idx.unsqueeze(-1) + torch.arange(sequence_length + extra, device=self.device)
            yield idx
            start_idx = start_idx[batch_size:]

    def compute_visit_entropy(self):
        if self.size <= 1:
            return 1.0
        # compute normalized entropy
        visits = self.sample_visits[:self.size]
        visit_sum = visits.sum()
        if visit_sum == 0:
            return 1.0
        max_entropy = math.log(self.size)
        entropy = D.Categorical(probs=visits / visit_sum).entropy().item()
        normalized_entropy = round(min(entropy / max_entropy, 1), 5)
        return normalized_entropy

    def _get_histogram(self, values, step):
        import wandb
        num_bins = int(math.ceil(self.size / step)) + 1
        bins = np.arange(num_bins) * step
        values = [v.sum().item() for v in torch.split(values, step)]
        return wandb.Histogram(np_histogram=[values, bins])

    def visit_histogram(self):
        visits = self.sample_visits[:self.size]
        return self._get_histogram(visits, step=500)

    def sample_probs_histogram(self):
        n = self.size
        visit_probs = self._compute_visit_probs(n)
        return self._get_histogram(visit_probs, step=500)

    def metrics(self):
        num_episodes = len(self.episode_lengths)
        metrics = {'size': self.size, 'total_reward': self.total_reward, 'num_episodes': num_episodes,
                   'visit_ent': self.compute_visit_entropy()}

        if num_episodes > self.metrics_num_episodes:
            new_episodes = num_episodes - self.metrics_num_episodes
            self.metrics_num_episodes = num_episodes
            metrics.update({'episode_len': utils.compute_mean(self.episode_lengths[-new_episodes:]),
                            'episode_score': np.mean(self.scores[-new_episodes:])})
        return metrics
