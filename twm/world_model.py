import math

import torch
import torch.nn.functional as F
import torch.distributions as D
from torch import nn
from torch.distributions.utils import logits_to_probs

import nets
import utils


class WorldModel(nn.Module):

    def __init__(self, config, num_actions):
        super().__init__()
        self.config = config
        self.num_actions = num_actions

        self.obs_model = ObservationModel(config)
        self.dyn_model = DynamicsModel(config, self.obs_model.z_dim, num_actions)

        self.obs_optimizer = utils.AdamOptim(
            self.obs_model.parameters(), lr=config['obs_lr'], eps=config['obs_eps'], weight_decay=config['obs_wd'],
            grad_clip=config['obs_grad_clip'])
        self.dyn_optimizer = utils.AdamOptim(
            self.dyn_model.parameters(), lr=config['dyn_lr'], eps=config['dyn_eps'], weight_decay=config['dyn_wd'],
            grad_clip=config['dyn_grad_clip'])

    @property
    def z_dim(self):
        return self.obs_model.z_dim

    @property
    def h_dim(self):
        return self.dyn_model.h_dim

    def optimize_pretrain_obs(self, o):
        obs_model = self.obs_model
        obs_model.train()

        z_dist = obs_model.encode(o)
        z = obs_model.sample_z(z_dist, reparameterized=True)
        recons = obs_model.decode(z)

        # no consistency loss required for pretraining
        dec_loss, dec_met = obs_model.compute_decoder_loss(recons, o)
        ent_loss, ent_met = obs_model.compute_entropy_loss(z_dist)

        obs_loss = dec_loss + ent_loss
        self.obs_optimizer.step(obs_loss)

        metrics = utils.combine_metrics([ent_met, dec_met])
        metrics['obs_loss'] = obs_loss.detach()
        return metrics

    def optimize_pretrain_dyn(self, z, a, r, terminated, truncated, target_logits):
        assert utils.same_batch_shape([z, a, r, terminated, truncated])
        assert utils.same_batch_shape_time_offset(z, target_logits, 1)
        dyn_model = self.dyn_model
        dyn_model.train()

        d = torch.logical_or(terminated, truncated)
        g = self.to_discounts(terminated)
        target_weights = (~d[:, 1:]).float()
        tgt_length = target_logits.shape[1]

        preds, h, mems = dyn_model.predict(z, a, r[:, :-1], g[:, :-1], d[:, :-1], tgt_length, compute_consistency=True)
        dyn_loss, metrics = dyn_model.compute_dynamics_loss(
            preds, h, target_logits=target_logits, target_r=r[:, 1:], target_g=g[:, 1:], target_weights=target_weights)
        self.dyn_optimizer.step(dyn_loss)
        return metrics

    def optimize(self, o, a, r, terminated, truncated):
        assert utils.same_batch_shape([a, r, terminated, truncated])
        assert utils.same_batch_shape_time_offset(o, r, 1)

        obs_model = self.obs_model
        dyn_model = self.dyn_model

        self.eval()
        with torch.no_grad():
            context_z_dist = obs_model.encode(o[:, :1])
            context_z = obs_model.sample_z(context_z_dist)
            next_z_dist = obs_model.encode(o[:, -1:])
            next_logits = next_z_dist.base_dist.logits

        self.train()

        # observation model
        o = o[:, 1:-1]
        z_dist = obs_model.encode(o)
        z = obs_model.sample_z(z_dist, reparameterized=True)
        recons = obs_model.decode(z)

        dec_loss, dec_met = obs_model.compute_decoder_loss(recons, o)
        ent_loss, ent_met = obs_model.compute_entropy_loss(z_dist)

        # dynamics model
        z = z.detach()
        z = torch.cat([context_z, z], dim=1)
        z_logits = z_dist.base_dist.logits
        target_logits = torch.cat([z_logits[:, 1:].detach(), next_logits.detach()], dim=1)
        d = torch.logical_or(terminated, truncated)
        g = self.to_discounts(terminated)
        target_weights = (~d[:, 1:]).float()
        tgt_length = target_logits.shape[1]

        preds, h, mems = dyn_model.predict(z, a, r[:, :-1], g[:, :-1], d[:, :-1], tgt_length, compute_consistency=True)
        dyn_loss, dyn_met = dyn_model.compute_dynamics_loss(
            preds, h, target_logits=target_logits, target_r=r[:, 1:], target_g=g[:, 1:], target_weights=target_weights)
        self.dyn_optimizer.step(dyn_loss)

        z_hat_probs = preds['z_hat_probs'].detach()
        con_loss, con_met = obs_model.compute_consistency_loss(z_logits, z_hat_probs)

        obs_loss = dec_loss + ent_loss + con_loss
        self.obs_optimizer.step(obs_loss)

        metrics = utils.combine_metrics([dec_met, ent_met, con_met, dyn_met])
        metrics['obs_loss'] = obs_loss.detach()

        return z, h, metrics

    @torch.no_grad()
    def to_discounts(self, mask):
        assert utils.check_no_grad(mask)
        discount_factor = self.config['env_discount_factor']
        g = torch.full(mask.shape, discount_factor, device=mask.device)
        g = g * (~mask).float()
        return g


class ObservationModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.z_dim = config['z_categoricals'] * config['z_categories']

        h = config['obs_channels']
        activation = config['obs_act']
        norm = config['obs_norm']
        dropout_p = config['obs_dropout']

        num_channels = config['env_frame_stack']
        if not config['env_grayscale']:
            num_channels *= 3

        self.encoder = nn.Sequential(
            nets.CNN(num_channels, [h, h * 2, h * 4], h * 8,
                     [4, 4, 4, 4], [2, 2, 2, 2], [0, 0, 0, 0], activation, norm=norm, post_activation=True),
            nn.Flatten(),
            nets.MLP((h * 8) * 2 * 2, [512, 512], self.z_dim, activation, norm=norm, dropout_p=dropout_p)
        )

        # no norm here
        self.decoder = nn.Sequential(
            nets.MLP(self.z_dim, [], (h * 16) * 1 * 1, activation, dropout_p=dropout_p, post_activation=True),
            nn.Unflatten(1, (h * 16, 1, 1)),
            nets.TransposeCNN(h * 16, [h * 4, h * 2, h], num_channels, [5, 5, 6, 6], [2, 2, 2, 2], [0, 0, 0, 0],
                              activation, final_bias_init=0.5)
        )

    @staticmethod
    def create_z_dist(logits, temperature=1):
        assert temperature > 0
        return D.Independent(D.OneHotCategoricalStraightThrough(logits=logits / temperature), 1)

    def encode(self, o):
        assert utils.check_no_grad(o)
        config = self.config
        shape = o.shape[:2]
        o = o.flatten(0, 1)

        if not config['env_grayscale']:
            o = o.permute(0, 1, 4, 2, 3)
            o = o.flatten(1, 2)

        z_logits = self.encoder(o)
        z_logits = z_logits.unflatten(0, shape)
        z_logits = z_logits.unflatten(-1, (config['z_categoricals'], config['z_categories']))
        z_dist = ObservationModel.create_z_dist(z_logits)
        return z_dist

    def sample_z(self, z_dist, reparameterized=False, temperature=1, idx=None, return_logits=False):
        logits = z_dist.base_dist.logits
        assert (not reparameterized) == utils.check_no_grad(logits)
        if temperature == 0:
            assert not reparameterized
            with torch.no_grad():
                if idx is not None:
                    logits = logits[idx]
                indices = torch.argmax(logits, dim=-1)
                z = F.one_hot(indices, num_classes=self.config['z_categories']).flatten(2, 3).float()
            if return_logits:
                return z, logits  # actually wrong logits for temperature = 0
            return z

        if temperature != 1 or idx is not None:
            if idx is not None:
                logits = logits[idx]
            z_dist = ObservationModel.create_z_dist(logits, temperature)
            if return_logits:
                logits = z_dist.base_dist.logits  # return new normalized logits

        z = z_dist.rsample() if reparameterized else z_dist.sample()
        z = z.flatten(2, 3)
        if return_logits:
            return z, logits
        return z

    def encode_sample(self, o, reparameterized=False, temperature=1, idx=None, return_logits=False):
        z_dist = self.encode(o)
        return self.sample_z(z_dist, reparameterized, temperature, idx, return_logits)

    def decode(self, z):
        config = self.config
        shape = z.shape[:2]
        z = z.flatten(0, 1)
        recons = self.decoder(z)
        if not config['env_grayscale']:
            recons = recons.unflatten(1, (config['env_frame_stack'], 3))
            recons = recons.permute(0, 1, 3, 4, 2)
        recons = recons.unflatten(0, shape)
        return recons

    def compute_decoder_loss(self, recons, o):
        assert utils.check_no_grad(o)
        config = self.config
        metrics = {}
        recon_mean = recons.flatten(0, 1).permute(0, 2, 3, 1)
        coef = config['obs_decoder_coef']
        if coef != 0:
            if config['env_grayscale']:
                o = o.flatten(0, 1).permute(0, 2, 3, 1)
            else:
                o = o.flatten(0, 1).permute(0, 2, 3, 1, 4).flatten(-2, -1)
            recon_dist = D.Independent(D.Normal(recon_mean, torch.ones_like(recon_mean)), 3)
            loss = -coef * recon_dist.log_prob(o).mean()
            metrics['dec_loss'] = loss.detach()
        else:
            loss = torch.zeros(1, device=recons.device, requires_grad=False)
        metrics['recon_mae'] = torch.abs(o - recon_mean.detach()).mean()
        return loss, metrics

    def compute_entropy_loss(self, z_dist):
        config = self.config
        metrics = {}

        entropy = z_dist.entropy().mean()
        max_entropy = config['z_categoricals'] * math.log(config['z_categories'])
        normalized_entropy = entropy / max_entropy
        metrics['z_ent'] = entropy.detach()
        metrics['z_norm_ent'] = normalized_entropy.detach()

        coef = config['obs_entropy_coef']
        if coef != 0:
            if config['obs_entropy_threshold'] < 1:
                # hinge loss, inspired by https://openreview.net/pdf?id=HkCjNI5ex
                loss = coef * torch.relu(config['obs_entropy_threshold'] - normalized_entropy)
            else:
                loss = -coef * normalized_entropy
            metrics['z_entropy_loss'] = loss.detach()
        else:
            loss = torch.zeros(1, device=z_dist.base_dist.logits.device, requires_grad=False)

        return loss, metrics

    def compute_consistency_loss(self, z_logits, z_hat_probs):
        assert utils.check_no_grad(z_hat_probs)
        config = self.config
        metrics = {}
        coef = config['obs_consistency_coef']
        if coef > 0:
            cross_entropy = -((z_hat_probs.detach() * z_logits).sum(-1))
            cross_entropy = cross_entropy.sum(-1)  # independent
            loss = coef * cross_entropy.mean()
            metrics['enc_prior_ce'] = cross_entropy.detach().mean()
            metrics['enc_prior_loss'] = loss.detach()
        else:
            loss = torch.zeros(1, device=z_logits.device, requires_grad=False)
        return loss, metrics


class DynamicsModel(nn.Module):

    def __init__(self, config, z_dim, num_actions):
        super().__init__()
        self.config = config

        embeds = {
            'z': {'in_dim': z_dim, 'categorical': False},
            'a': {'in_dim': num_actions, 'categorical': True}
        }
        modality_order = ['z', 'a']
        num_current = 2

        if config['dyn_input_rewards']:
            embeds['r'] = {'in_dim': 0, 'categorical': False}
            modality_order.append('r')

        if config['dyn_input_discounts']:
            embeds['g'] = {'in_dim': 0, 'categorical': False}
            modality_order.append('g')

        self.modality_order = modality_order

        out_heads = {
            'z': {'hidden_dims': config['dyn_z_dims'], 'out_dim': z_dim},
            'r': {'hidden_dims': config['dyn_reward_dims'], 'out_dim': 1, 'final_bias_init': 0.0},
            'g': {'hidden_dims': config['dyn_discount_dims'], 'out_dim': 1,
                  'final_bias_init': config['env_discount_factor']}
        }

        memory_length = config['wm_memory_length']
        max_length = 1 + config['wm_sequence_length']  # 1 for context
        self.prediction_net = nets.PredictionNet(
            modality_order, num_current, embeds, out_heads, embed_dim=config['dyn_embed_dim'],
            activation=config['dyn_act'], norm=config['dyn_norm'], dropout_p=config['dyn_dropout'],
            feedforward_dim=config['dyn_feedforward_dim'], head_dim=config['dyn_head_dim'],
            num_heads=config['dyn_num_heads'], num_layers=config['dyn_num_layers'],
            memory_length=memory_length, max_length=max_length)

    @property
    def h_dim(self):
        return self.prediction_net.embed_dim

    def predict(self, z, a, r, g, d, tgt_length, heads=None, mems=None, return_attention=False,
                compute_consistency=False):
        assert utils.check_no_grad(z, a, r, g, d)
        assert mems is None or utils.check_no_grad(*mems)
        config = self.config

        if compute_consistency:
            tgt_length += 1  # add 1 timestep for context

        inputs = {'z': z, 'a': a, 'r': r, 'g': g}
        heads = tuple(heads) if heads is not None else ('z', 'r', 'g')

        outputs = self.prediction_net(
            inputs, tgt_length, stop_mask=d, heads=heads, mems=mems, return_attention=return_attention)
        out, h, mems, attention = outputs if return_attention else (outputs + (None,))

        preds = {}

        if 'z' in heads:  # latent states
            z_categoricals = config['z_categoricals']
            z_categories = config['z_categories']
            z_logits = out['z'].unflatten(-1, (z_categoricals, z_categories))

            if compute_consistency:
                # used for consistency loss
                preds['z_hat_probs'] = ObservationModel.create_z_dist(z_logits[:, :-1].detach()).base_dist.probs
                z_logits = z_logits[:, 1:]  # remove context

            z_dist = ObservationModel.create_z_dist(z_logits)
            preds['z_dist'] = z_dist

        if 'r' in heads:  # rewards
            r_params = out['r']
            if compute_consistency:
                r_params = r_params[:, 1:]  # remove context
            r_mean = r_params.squeeze(-1)
            r_dist = D.Normal(r_mean, torch.ones_like(r_mean))

            r_pred = r_dist.mean
            preds['r_dist'] = r_dist  # used for dynamics loss
            preds['r'] = r_pred

        if 'g' in heads:  # discounts
            g_params = out['g']
            if compute_consistency:
                g_params = g_params[:, 1:]  # remove context
            g_mean = g_params.squeeze(-1)
            g_dist = D.Bernoulli(logits=g_mean)

            g_pred = torch.clip(g_dist.mean, 0, 1)
            preds['g_dist'] = g_dist  # used for dynamics loss
            preds['g'] = g_pred

        return (preds, h, mems, attention) if return_attention else (preds, h, mems)

    def compute_dynamics_loss(self, preds, h, target_logits, target_r, target_g, target_weights):
        assert utils.check_no_grad(target_logits, target_r, target_g, target_weights)
        config = self.config
        losses = []
        metrics = {}

        metrics['h_norm'] = h.norm(dim=-1, p=2).mean().detach()

        if 'z' in preds:
            z_dist = preds['z_dist']
            z_logits = z_dist.base_dist.logits  # use normalized logits

            # doesn't check for q == 0
            target_probs = logits_to_probs(target_logits)
            cross_entropy = -((target_probs * z_logits).sum(-1))
            cross_entropy = cross_entropy.sum(-1)  # independent
            weighted_cross_entropy = target_weights * cross_entropy
            weighted_cross_entropy = weighted_cross_entropy.sum() / target_weights.sum()

            coef = config['dyn_z_coef']
            if coef != 0:
                transition_loss = coef * weighted_cross_entropy
                losses.append(transition_loss)

                metrics['z_pred_loss'] = transition_loss.detach()
                metrics['z_pred_ent'] = z_dist.entropy().detach().mean()
                metrics['z_pred_ce'] = weighted_cross_entropy.detach()

            # doesn't check for q == 0
            kl = (target_probs * (target_logits - z_logits.detach())).mean()
            kl = F.relu(kl.mean())
            metrics['z_kl'] = kl

        if 'r' in preds:
            r_dist = preds['r_dist']
            r_pred = preds['r']
            coef = config['dyn_reward_coef']
            if coef != 0:
                r_loss = -coef * r_dist.log_prob(target_r).mean()
                losses.append(r_loss)
                metrics['reward_loss'] = r_loss.detach()
                metrics['reward_mae'] = torch.abs(target_r - r_pred.detach()).mean()
            metrics['reward'] = r_pred.mean().detach()

        if 'g' in preds:
            g_dist = preds['g_dist']
            g_pred = preds['g']
            coef = config['dyn_discount_coef']
            if coef != 0:
                g_dist._validate_args = False
                g_loss = -coef * g_dist.log_prob(target_g).mean()
                losses.append(g_loss)
                metrics['discount_loss'] = g_loss.detach()
                metrics['discount_mae'] = torch.abs(target_g - g_pred.detach()).mean()
            metrics['discount'] = g_pred.detach().mean()

        if len(losses) == 0:
            loss = torch.zeros(1, device=z.device, requires_grad=False)
        else:
            loss = sum(losses)
            metrics['dyn_loss'] = loss.detach()
        return loss, metrics
