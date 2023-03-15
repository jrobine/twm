import copy
import math

import torch
import torch.distributions as D
from torch import nn

import nets
import utils


class ActorCritic(nn.Module):

    def __init__(self, config, num_actions, z_dim, h_dim):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        activation = config['ac_act']
        norm = config['ac_norm']
        dropout_p = config['ac_dropout']

        input_dim = z_dim
        if config['ac_input_h']:
            input_dim += h_dim

        self.h_norm = nets.get_norm_1d(config['ac_h_norm'], h_dim)
        self.trunk = nn.Identity()
        self.actor_model = nets.MLP(
            input_dim, config['actor_dims'], num_actions, activation, norm=norm, dropout_p=dropout_p,
            weight_initializer='orthogonal', bias_initializer='zeros')
        self.critic_model = nets.MLP(
            input_dim, config['critic_dims'], 1, activation, norm=norm, dropout_p=dropout_p,
            weight_initializer='orthogonal', bias_initializer='zeros')
        if config['critic_target_interval'] > 1:
            self.target_critic_model = copy.deepcopy(self.critic_model).requires_grad_(False)
            self.register_buffer('target_critic_lag', torch.zeros(1, dtype=torch.long))

        self.actor_optimizer = utils.AdamOptim(
            self.actor_model.parameters(), lr=config['actor_lr'], eps=config['actor_eps'],
            weight_decay=config['actor_wd'], grad_clip=config['actor_grad_clip'])
        self.critic_optimizer = utils.AdamOptim(
            self.critic_model.parameters(), lr=config['critic_lr'], eps=config['critic_eps'],
            weight_decay=config['critic_wd'], grad_clip=config['critic_grad_clip'])

        self.sync_target()

    @torch.no_grad()
    def _prepare_inputs(self, z, h):
        assert utils.check_no_grad(z, h)
        assert h is None or utils.same_batch_shape([z, h])
        config = self.config
        if config['ac_input_h']:
            h = self.h_norm(h)
            x = torch.cat([z, h], dim=-1)
        else:
            x = z
        shape = x.shape[:2]
        x = self.trunk(x.flatten(0, 1)).unflatten(0, shape)
        return x

    def actor(self, x):
        shape = x.shape[:2]
        logits = self.actor_model(x.flatten(0, 1)).unflatten(0, shape)
        return logits

    def critic(self, x):
        shape = x.shape[:2]
        values = self.critic_model(x.flatten(0, 1)).squeeze(-1).unflatten(0, shape)
        return values

    def sync_target(self):
        if self.config['critic_target_interval'] > 1:
            self.target_critic_lag[:] = 0
            self.target_critic_model.load_state_dict(self.critic_model.state_dict())

    def optimize(self, z, h, a, r, g, d, weights):
        x = self._prepare_inputs(z, h)
        returns, advantages = self._compute_targets(x, r, g, d)
        self.train()

        # remove last time step, the last state is for bootstrapping
        values = self.critic(x[:, :-1])
        critic_loss, critic_metrics = self._compute_critic_loss(values, returns, weights)
        self.critic_optimizer.step(critic_loss)

        logits = self.actor(x[:, :-1])
        actor_loss, actor_metrics = self._compute_actor_loss(logits, a, advantages, weights)
        self.actor_optimizer.step(actor_loss)

        metrics = utils.combine_metrics([critic_metrics, actor_metrics])
        if d is not None:
            metrics['num_dones'] = d.sum().detach()  # number of imagined dones
        return metrics

    def optimize_pretrain(self, z, h, r, g, d):
        config = self.config
        x = self._prepare_inputs(z, h)
        returns, advantages = self._compute_targets(x, r, g, d)
        weights = torch.ones_like(returns)  # no weights, since we use real data

        self.train()
        # remove last time step, the last state is for bootstrapping
        values = self.critic(x[:, :-1])
        critic_loss, critic_metrics = self._compute_critic_loss(values, returns, weights)

        # maximize entropy, ok since data was collected with random policy
        shape = x.shape[:2]
        logits = self.actor_model(x.flatten(0, 1)).unflatten(0, shape)
        dist = D.Categorical(logits=logits)
        max_entropy = math.log(self.num_actions)
        entropy = dist.entropy().mean()
        normalized_entropy = entropy / max_entropy
        actor_loss = -config['actor_entropy_coef'] * normalized_entropy
        actor_metrics = {
            'actor_loss': actor_loss.detach(), 'ent': entropy.detach(), 'norm_ent': normalized_entropy.detach()
        }

        self.actor_optimizer.step(actor_loss)
        self.critic_optimizer.step(critic_loss)

        return utils.combine_metrics([critic_metrics, actor_metrics])

    def _compute_actor_loss(self, logits, a, advantages, weights):
        assert utils.check_no_grad(a, advantages, weights)
        config = self.config
        dist = D.Categorical(logits=logits)
        reinforce = dist.log_prob(a) * advantages
        reinforce = (weights * reinforce).mean()
        loss = -reinforce

        entropy = weights * dist.entropy()
        max_entropy = math.log(self.num_actions)
        normalized_entropy = (entropy / max_entropy).mean()
        coef = config['actor_entropy_coef']
        if coef != 0:
            entropy_reg = coef * torch.relu(config['actor_entropy_threshold'] - normalized_entropy)
            loss = loss + entropy_reg

        metrics = {
            'actor_loss': loss.detach(), 'reinforce': reinforce.detach().mean(), 'ent': entropy.detach().mean(),
            'norm_ent': normalized_entropy.detach()
        }
        return loss, metrics

    def _compute_critic_loss(self, values, returns, weights):
        assert utils.check_no_grad(returns, weights)
        value_dist = D.Normal(values, torch.ones_like(values))
        loss = -(weights * value_dist.log_prob(returns)).mean()
        mae = torch.abs(returns - values.detach()).mean()
        metrics = {'critic_loss': loss.detach(), 'critic_mae': mae, 'critic': values.detach().mean(),
                   'returns': returns.mean()}
        return loss, metrics

    @torch.no_grad()
    def _compute_gae(self, r, g, values, dones=None):
        assert utils.same_batch_shape([r, g])
        assert dones is None or utils.same_batch_shape([r, dones])
        assert utils.same_batch_shape_time_offset(values, r, 1)
        assert utils.check_no_grad(r, g, values, dones)
        stopped_discounts = (g * (~dones).float()) if dones is not None else discounts
        delta = r + stopped_discounts * values[:, 1:] - values[:, :-1]
        advantages = torch.zeros_like(values)
        factors = stopped_discounts * self.config['env_discount_lambda']
        for t in range(r.shape[1] - 1, -1, -1):
            advantages[:, t] = delta[:, t] + factors[:, t] * advantages[:, t + 1]
        advantages = advantages[:, :-1]
        return advantages

    @torch.no_grad()
    def _compute_targets(self, x, r, g, d=None):
        # adopted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py
        assert utils.same_batch_shape([r, g])
        assert utils.same_batch_shape_time_offset(x, r, 1)
        assert d is None or utils.same_batch_shape([r, d])
        assert utils.check_no_grad(x, r, g, d)
        config = self.config
        self.eval()

        shape = x.shape[:2]
        if config['critic_target_interval'] > 1:
            self.target_critic_lag += 1
            if self.target_critic_lag >= config['critic_target_interval']:
                self.sync_target()
            values = self.target_critic_model(x.flatten(0, 1)).squeeze(-1).unflatten(0, shape)
        else:
            values = self.critic_model(x.flatten(0, 1)).squeeze(-1).unflatten(0, shape)

        advantages = self._compute_gae(r, g, values, d)
        returns = advantages + values[:, :-1]
        if config['ac_normalize_advantages']:
            adv_mean = advantages.mean()
            adv_std = torch.std(advantages, unbiased=False)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        return returns, advantages

    @torch.no_grad()
    def policy(self, z, h, temperature=1):
        assert utils.check_no_grad(z, h)
        self.eval()
        x = self._prepare_inputs(z, h)
        logits = self.actor(x)

        if temperature == 0:
            actions = logits.argmax(dim=-1)
        else:
            if temperature != 1:
                logits = logits / temperature
            actions = D.Categorical(logits=logits / temperature).sample()
        return actions
