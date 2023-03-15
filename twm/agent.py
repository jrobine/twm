import torch
from torch import nn

from actor_critic import ActorCritic
from world_model import WorldModel
import utils


class Agent(nn.Module):

    def __init__(self, config, num_actions):
        super().__init__()
        self.config = config
        self.wm = WorldModel(config, num_actions)
        self.ac = ActorCritic(config, num_actions, self.wm.z_dim, self.wm.h_dim)


class Dreamer:
    # reset: s_t-1, a_t-1, r_t-1, d_t-1, s_t => s_t, h_t-1
    # step:  a_t => s_t+1, h_t, r_t, d_t

    def __init__(self, config, wm, mode, ac=None, store_data=False, start_z_sampler=None, always_compute_obs=False):
        assert mode in ('imagine', 'observe')
        assert mode != 'imagine' or start_z_sampler is not None
        self.config = config
        self.wm = wm
        self.ac = ac
        self.mode = mode
        self.store_data = store_data
        self.start_z_sampler = start_z_sampler
        self.always_compute_obs = always_compute_obs

        self.cumulative_g = None  # cumulative discounts
        self.stop_mask = None  # history of dones, for transformer
        self.mems = None
        self.prev_z = None
        self.prev_o = None
        self.prev_h = None
        self.prev_r = None
        self.prev_g = None  # discounts
        self.prev_d = None  # episode ends

        if store_data:
            self.z_data = None
            self.o_data = None
            self.h_data = None
            self.a_data = None
            self.r_data = None
            self.g_data = None
            self.d_data = None
            self.weight_data = None

    @torch.no_grad()
    def get_data(self):
        assert self.store_data
        z = torch.cat(self.z_data, dim=1)
        o = torch.cat(self.o_data, dim=1) if len(self.o_data) > 0 else None
        h = torch.cat(self.h_data, dim=1)
        a = torch.cat(self.a_data, dim=1)
        r = torch.cat(self.r_data, dim=1)
        g = torch.cat(self.g_data, dim=1)
        d = torch.cat(self.d_data, dim=1)
        weights = torch.cat(self.weight_data, dim=1)
        return z, o, h, a, r, g, d, weights

    def _zero_h(self, batch_size, device):
        return torch.zeros(batch_size, 1, self.wm.h_dim, device=device)

    def _reset(self, start_z, start_a, start_r, start_terminated, start_truncated, keep_start_data=False):
        assert utils.same_batch_shape([start_a, start_r, start_terminated, start_truncated])
        assert utils.same_batch_shape_time_offset(start_z, start_r, 1)
        assert not (keep_start_data and not self.store_data)
        config = self.config
        wm = self.wm.eval()
        obs_model = wm.obs_model
        dyn_model = wm.dyn_model

        start_g = wm.to_discounts(start_terminated)
        start_d = torch.logical_or(start_terminated, start_truncated)
        if self.mode == 'imagine' or (self.mode == 'observe' and config['ac_input_h']):
            if start_a.shape[1] == 0:
                h = self._zero_h(start_a.shape[0], start_a.device)
                mems = None
            else:
                _, h, mems = dyn_model.predict(
                    start_z[:, :-1], start_a, start_r[:, :-1], start_g[:, :-1], start_d[:, :-1], heads=[], tgt_length=1)
        else:
            h, mems = None, None

        # set cumulative_g to 1 for real data, start discounting after that
        start_weights = (~start_d).float()
        self.cumulative_g = torch.ones_like(start_g[:, -1:])
        self.stop_mask = start_d

        z = start_z[:, -1:]
        r = start_r[:, -1:]
        g = start_g[:, -1:]
        d = start_d[:, -1:]

        self.mems = mems
        self.prev_z = z
        self.prev_h = h
        self.prev_r = r
        self.prev_g = g
        self.prev_d = d

        if self.store_data:
            self.h_data = [self._zero_h(start_z.shape[0], start_z.device) if h is None else h]

            if keep_start_data:
                self.z_data = [start_z]
                self.a_data = [start_a]
                self.r_data = [start_r]
                self.g_data = [start_g]
                self.d_data = [start_d]
                self.weight_data = [start_weights]
            else:
                self.z_data = [z]
                self.a_data = []
                self.r_data = []
                self.g_data = []
                self.d_data = []
                self.weight_data = []

        if self.always_compute_obs:
            start_o = obs_model.decode(start_z)
            o = start_o[:, -1:]
            self.prev_o = o
            if self.store_data:
                if keep_start_data:
                    self.o_data = [start_o]
                else:
                    self.o_data = [o]
        else:
            if self.store_data:
                self.o_data = []

        return z, h, start_g, start_d

    @torch.no_grad()
    def imagine_reset(self, start_z, start_a, start_r, start_terminated, start_truncated, keep_start_data=False):
        assert self.mode == 'imagine'
        # returns: z, h, start_g, start_d
        return self._reset(start_z, start_a, start_r, start_terminated, start_truncated, keep_start_data)

    @torch.no_grad()
    def observe_reset(self, start_o, start_a, start_r, start_terminated, start_truncated, keep_start_data=False):
        assert self.mode == 'observe'
        obs_model = self.wm.obs_model.eval()
        start_z = obs_model.encode_sample(start_o, temperature=0)
        z, h, start_g, start_d = self._reset(
            start_z, start_a, start_r, start_terminated, start_truncated, keep_start_data)
        return z, h, start_z, start_g, start_d

    @staticmethod
    def _create_single_data(batch_size, device):
        start_a = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        start_r = torch.zeros(batch_size, 0, device=device)
        start_terminated = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        start_truncated = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        return start_a, start_r, start_terminated, start_truncated

    @torch.no_grad()
    def imagine_reset_single(self, start_z, keep_start_data=False):
        assert start_z.shape[1] == 1
        start_a, start_r, start_terminated, start_truncated = self._create_single_data(start_z.shape[0], start_z.device)
        return self.imagine_reset(start_z, start_a, start_r, start_terminated, start_truncated, keep_start_data)

    @torch.no_grad()
    def observe_reset_single(self, start_o, keep_start_data=False):
        assert start_o.shape[1] == 1
        start_a, start_r, start_terminated, start_truncated = self._create_single_data(start_o.shape[0], start_o.device)
        return self.observe_reset(start_o, start_a, start_r, start_terminated, start_truncated, keep_start_data)

    def _step(self, a, z, r, g, d, temperature, return_attention):
        config = self.config
        imagine = self.mode == 'imagine'
        assert a.shape[1] == 1
        assert all(x is None for x in (z, r, g, d)) if imagine else utils.same_batch_shape([a, z, r, g, d])
        wm = self.wm.eval()
        obs_model = wm.obs_model
        dyn_model = wm.dyn_model

        z_dist = None
        if imagine or self.config['ac_input_h']:
            assert self.mems is not None or self.prev_r.shape[1] == 0
            assert self.mems is None or a.shape[0] == self.mems[0].shape[1]
            heads = ['z', 'r', 'g'] if imagine else []
            outputs = dyn_model.predict(
                self.prev_z, a, self.prev_r, self.prev_g, self.stop_mask, tgt_length=1, heads=heads, mems=self.mems,
                return_attention=return_attention)
            preds, h, mems, attention = outputs if return_attention else (outputs + (None,))
            if imagine:
                z_dist = preds['z_dist']
                z = obs_model.sample_z(z_dist, temperature=temperature)
                r = preds['r']
                g = preds['g']
        else:
            h, mems, attention = None, None, None

        if self.cumulative_g.shape[1] == 0:
            weights = torch.ones_like(g)
            self.cumulative_g = g.clone()
        else:
            done = self.prev_d.float()
            not_done = (~self.prev_d).float()
            weights = self.cumulative_g * not_done + torch.ones_like(self.prev_g) * done
            self.cumulative_g = (not_done * self.cumulative_g + done) * g

        if imagine:
            if config['wm_discount_threshold'] > 0:
                d = (self.cumulative_g < config['wm_discount_threshold'])
                num_done = d.sum()
                if num_done > 0:
                    new_start_z = self.start_z_sampler(num_done)
                    z[d] = new_start_z
            else:
                d = torch.zeros(a.shape[0], 1, dtype=torch.bool, device=a.device)

        stop_mask = torch.cat([self.stop_mask, d], dim=1)
        memory_length = config['wm_memory_length']
        if stop_mask.shape[1] > memory_length + 1:
            stop_mask = stop_mask[:, -(memory_length + 1):]
        self.stop_mask = stop_mask

        self.mems = mems
        self.prev_z, self.prev_h, self.prev_r, self.prev_g, self.prev_d = z, h, r, g, d

        if self.store_data:
            self.z_data.append(z)
            self.h_data.append(h)
            self.a_data.append(a)
            self.r_data.append(r)
            self.g_data.append(g)
            self.d_data.append(d)
            self.weight_data.append(weights)

        if self.always_compute_obs:
            o = obs_model.decode(z)
            self.prev_o = o
            if self.store_data:
                self.o_data.append(o)

        outputs = (z, h, z_dist, r, g, d, weights)
        if return_attention:
            outputs = outputs + (attention,)
        return outputs

    @torch.no_grad()
    def imagine_step(self, a, temperature=1, return_attention=False):
        assert self.mode == 'imagine'
        # returns: z, h, z_dist, r, g, d, weights, [attention]
        return self._step(a, None, None, None, None, temperature, return_attention)

    @torch.no_grad()
    def observe_step(self, a, o, r, terminated, truncated, return_attention=False):
        assert self.mode == 'observe'
        wm = self.wm
        obs_model = wm.obs_model
        obs_model.eval()
        z = obs_model.encode_sample(o, temperature=0)
        g = wm.to_discounts(terminated)
        d = torch.logical_or(terminated, truncated)
        if return_attention:
            _, h, _, _, _, _, weights, attention = self._step(a, z, r, g, d, temperature=None, return_attention=True)
            return z, h, g, d, weights, attention
        else:
            _, h, _, _, _, _, weights = self._step(a, z, r, g, d, temperature=None, return_attention=False)
            return z, h, g, d, weights

    @torch.no_grad()
    def act(self, temperature=1, epsilon=0):
        z, h = self.prev_z, self.prev_h
        a = self.ac.policy(z, h, temperature=temperature)
        if epsilon > 0:
            num_actions = self.ac.num_actions
            epsilon_mask = torch.rand_like(a, dtype=torch.float) < epsilon
            random_actions = torch.randint_like(a, num_actions)
            a[epsilon_mask] = random_actions[epsilon_mask]
        return a
