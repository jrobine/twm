import argparse
from PIL import Image, ImageDraw

import gym
import numpy as np
import torch

from agent import Agent, Dreamer
import utils


class DreamEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}
    frame_stack_fns = {
        'last': lambda obs: obs[-1],
        'mean': lambda obs: np.mean(obs, axis=0),
        'max': lambda obs: np.max(obs, axis=0)
    }

    def __init__(self, config, wm, temperature=1, ac=None,
                 render_mode=None, render_frame_stack='last', render_extra=False):
        assert render_mode is None or render_mode in self.metadata['render_modes']
        assert render_frame_stack in self.frame_stack_fns
        super().__init__()
        self.config = config
        self.wm = wm
        self.temperature = temperature
        self.dreamer = Dreamer(
            config, wm, mode='imagine', ac=ac, store_data=False, start_z_sampler=self._start_z_sampler)
        self.prev_logits = None  # used for manipulation with the mouse
        self.attention_history = []
        self.action_history = []
        self.reward_history = []
        self.attention_mean = False

        z_dim = wm.z_dim
        h_dim = wm.h_dim
        z_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(z_dim,))
        h_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(h_dim,))
        self.observation_space = gym.spaces.Tuple([z_space, h_space])
        self.action_space = gym.spaces.Discrete(wm.num_actions)

        start_o_env = gym.make(f'{config["game"]}NoFrameskip-v4')
        start_o_env = gym.wrappers.AtariPreprocessing(
            start_o_env, noop_max=0, frame_skip=config['env_frame_skip'], screen_size=config['env_frame_size'],
            terminal_on_life_loss=False, grayscale_obs=config['env_grayscale'])
        start_o_env = gym.wrappers.FrameStack(start_o_env, config['env_frame_stack'])
        self.start_o_env = start_o_env
        self.action_meanings = start_o_env.get_action_meanings()

        self.render_mode = render_mode
        self.render_zoom = 3
        self.render_frame_stack = self.frame_stack_fns[render_frame_stack]
        self.render_extra = render_extra
        self.z_rect = (0, 0, 0, 0)

    def close(self):
        self.start_o_env.close()
        self.start_o_env = None

    def _start_z_sampler(self, n):
        obs_model = self.wm.obs_model.eval()
        if torch.is_tensor(n):
            n = n.item()
        o = np.array([self.start_o_env.reset()[0] for _ in range(n)])
        device = next(obs_model.parameters()).device
        o = utils.preprocess_atari_obs(o, device).unsqueeze(0).unsqueeze(1)
        with torch.no_grad():
            z = obs_model.encode_sample(o, temperature=0)
        return z

    @torch.no_grad()
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        obs_model = self.wm.obs_model.eval()
        device = next(obs_model.parameters()).device
        start_o, _ = self.start_o_env.reset(seed=seed)
        start_o = utils.preprocess_atari_obs(start_o, device).unsqueeze(0).unsqueeze(1)
        start_z, start_logits = obs_model.encode_sample(start_o, temperature=0, return_logits=True)
        z, h, _, _ = self.dreamer.imagine_reset_single(start_z)
        self.prev_logits = start_logits[:, -1:]
        self.attention_history = []
        self.action_history = []
        self.reward_history = []

        obs = (z.squeeze(0).squeeze(1), h.squeeze(0).squeeze(1))
        return obs, {}

    @torch.no_grad()
    def step(self, action):
        wm = self.wm
        device = next(wm.parameters()).device
        a = torch.as_tensor(action, device=device).unsqueeze(0).unsqueeze(1)
        z, h, z_dist, r, g, d, weights, attention = \
            self.dreamer.imagine_step(a, self.temperature, return_attention=True)

        obs = (z.squeeze(0).squeeze(1), h.squeeze(0).squeeze(1))
        reward = r.item()
        terminated = d.item()
        truncated = False
        info = {}

        self.prev_logits = z_dist.base_dist.logits
        attention = attention.squeeze(2).squeeze(0)
        attention = attention.mean(dim=-1)  # max over heads
        attention = attention.cpu().numpy()
        if len(self.attention_history) > 0 and attention.shape != self.attention_history[-1].shape:
            self.attention_history = []
        self.attention_history.append(attention)
        self.action_history.append(action)
        self.reward_history.append(reward)

        return obs, reward, terminated, truncated, info

    @torch.no_grad()
    def act(self, temperature=1, epsilon=0):
        return self.dreamer.act(temperature=temperature, epsilon=epsilon)

    @torch.no_grad()
    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array'
        config = self.config
        obs_model = self.wm.obs_model.eval()
        o = obs_model.decode(self.dreamer.prev_z)
        o = o.squeeze(1).squeeze(0).cpu().numpy()
        array = self.render_frame_stack(o)
        array = np.clip(array, 0, 1)

        if config['env_grayscale']:
            array = array[:, :, np.newaxis]
            array = np.repeat(array, 3, axis=2)

        obs_h, obs_w = array.shape[:2]

        extra_array = np.zeros(array.shape, dtype=array.dtype)
        y = 4

        if self.render_extra:
            # assumes representation is smaller than frame
            with torch.no_grad():
                z_probs = torch.nn.functional.softmax(self.prev_logits, dim=-1)
            z_array = z_probs.cpu().squeeze(1).squeeze(0).numpy()
            z_array = np.repeat(z_array[:, :, np.newaxis], 3, axis=2)

            x = (extra_array.shape[1] - z_array.shape[1]) // 2
            self.z_rect = (y, x + obs_w, y + z_array.shape[0], x + obs_w + z_array.shape[1])
            extra_array[y - 1:y + z_array.shape[0] + 1, x - 1:x + z_array.shape[1] + 1] = 1  # white border
            extra_array[y:y + z_array.shape[0], x:x + z_array.shape[1]] = z_array
            y += z_array.shape[0]
            y += 4

            modality_order = self.wm.dyn_model.modality_order
            num_modalities = len(modality_order)
            z_index = modality_order.index('z')
            a_index = modality_order.index('a')
            r_index = modality_order.index('r')
            max_timesteps = (config['wm_memory_length'] + 1) * num_modalities + 2
            x = (extra_array.shape[1] - max_timesteps) // 2 + 1
            if self.attention_mean:
                attention = np.mean(np.array(self.attention_history[-30:]), axis=0)
            else:
                attention = self.attention_history[-1]
            attention = attention / np.sum(attention, axis=0, keepdims=True)

            num_timesteps, num_layers = attention.shape
            # white border
            extra_array[y - 1:y + num_layers + (1 if self.attention_mean else 2), x - 1:x + max_timesteps + 1] = 1
            #extra_array[y:y + num_layers + (0 if self.attention_mean else 1), x:x + max_timesteps] = 0
            for t in range(num_timesteps):
                if t % num_modalities == z_index:
                    color = (33/255, 150/255, 243/255)
                elif t % num_modalities == a_index:
                    color = (244/255, 67/255, 54/255)
                elif t % num_modalities == r_index:
                    if self.attention_mean:
                        color = (76/255, 175/255, 80/255)
                    else:
                        reward = self.reward_history[(-num_timesteps + t) // num_modalities - 1]
                        if abs(reward) < 0.1:
                            color = (1, 1, 1)
                        elif reward > 0:
                            color = (76/255, 175/255, 80/255)
                        else:
                            color = (156/255, 39/255, 176/255)
                else:
                    color = (1, 1, 1)
                weights = attention[t][::-1, np.newaxis]
                weights = np.minimum(weights * 3, 1)
                weights = np.repeat(weights, 3, axis=1)
                attn_colors = (1 - weights) * np.ones((1, 3)) + weights * np.array([color])
                extra_array[y:y + num_layers, x + max_timesteps - num_timesteps + t] = attn_colors
                if not self.attention_mean:
                    extra_array[y + num_layers, x + max_timesteps - num_timesteps + t] = color
            y += num_layers
            y += 4

            array = np.concatenate((array, extra_array), axis=1)
        array = (array * 255).astype(np.uint8)

        zoom = self.render_zoom
        img = Image.fromarray(array, mode='RGB')
        img = img.resize((array.shape[1] * zoom, array.shape[0] * zoom), resample=0)

        if self.render_extra:
            # render action text
            draw = ImageDraw.Draw(img)
            x = obs_w * zoom + (extra_array.shape[1] * zoom - draw.textlength('action: UPRIGHT')) // 2
            draw.text((x, (y + 1) * zoom), f'action: {self.action_meanings[self.action_history[-1]][:7]: >7}')
            draw.text((x, (y + 4) * zoom), f'reward: {self.reward_history[-1]: .4f}')

        array = np.array(img)
        return array

    @torch.no_grad()
    def mouse_click(self, pos):
        if self.render_extra:
            # manipulate z with the mouse
            y, x = pos[0] // self.render_zoom, pos[1] // self.render_zoom
            t, l, b, r = self.z_rect
            if l <= x < r and t <= y < b:
                x -= l
                y -= t
                min_logit = self.prev_logits.min()
                max_logit = self.prev_logits.max()
                self.prev_logits[:, :, y] = min_logit
                self.prev_logits[:, :, y, x] = max_logit
                logits = self.prev_logits
                categories = self.config['z_categories']
                new_z = torch.nn.functional.one_hot(torch.argmax(logits, dim=-1), num_classes=categories)
                new_z = new_z.flatten(2, 3).float()
                self.dreamer.prev_z = new_z  # override prev z

    def press_toggle(self, key):
        import pygame
        if key == pygame.K_F3:
            self.attention_mean = not self.attention_mean


class RenderWrapper(gym.Wrapper):

    def __init__(self, env, zoom, fps, keys_to_action):
        super().__init__(env)
        #if env.render_mode != 'rgb_array':     # TODO AtariEnv does not return the render mode in gym 0.26.0
        #    raise ValueError(env.render_mode)
        self.fps = fps
        self.zoom = zoom
        self.screen = None
        self.clock = None
        self.screen_width = None
        self.screen_height = None

        self.relevant_keys = set(sum((list(k) for k in keys_to_action.keys()), []))
        self.pressed_keys = []
        self.mouse_down = False
        self.quit = False
        self.pause = True
        self.advance = False

        key_code_to_action = {}
        for key_combination, action in keys_to_action.items():
            key_code = tuple(sorted(ord(key) if isinstance(key, str) else key for key in key_combination))
            key_code_to_action[key_code] = action
        self.key_code_to_action = key_code_to_action

    def get_pressed_action(self):
        noop = 0
        return self.key_code_to_action.get(tuple(sorted(self.pressed_keys)), noop)

    def render(self):
        import pygame
        rgb_array = self.env.render()
        rgb_array = np.swapaxes(rgb_array, 0, 1)

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen_width = rgb_array.shape[0] * self.zoom
            self.screen_height = rgb_array.shape[1] * self.zoom
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        self.advance = False

        surf = pygame.surfarray.make_surface(rgb_array)
        surf = pygame.transform.scale(surf, (self.screen_width, self.screen_height))
        self.screen.blit(surf, (0, 0))
        pygame.event.pump()

        def mouse_click():
            pos = pygame.mouse.get_pos()
            self.mouse_click((pos[0] // self.zoom, pos[1] // self.zoom))

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in self.relevant_keys:
                    self.pressed_keys.append(event.key)
                elif event.key == pygame.K_ESCAPE:
                    self.quit = True
                elif event.key == pygame.K_F1:
                    self.pause = not self.pause
                elif event.key == pygame.K_F2:
                    if self.pause:
                        self.advance = True
                elif event.key == pygame.K_F3:
                    self.press_toggle(pygame.K_F3)
            elif event.type == pygame.KEYUP:
                if event.key in self.relevant_keys:
                    self.pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.VIDEORESIZE:
                video_size = event.size
                self.screen = pygame.display.set_mode(video_size)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_down = True
                mouse_click()
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_down = False
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_down:
                    mouse_click()

        self.clock.tick(self.fps)
        pygame.display.flip()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            self.pause = True
        self.render()
        return obs, reward, terminated, truncated, info

    def wait(self):
        while self.pause and not self.advance and not self.quit:
            self.render()


def play_real(config, state_dict, device, user_input, seed):
    zoom = 3
    fps = 60
    env = gym.make(f'{config["game"]}NoFrameskip-v4', render_mode='rgb_array')
    num_actions = env.action_space.n
    keys_to_action = env.get_keys_to_action()
    env = RenderWrapper(env, zoom, fps, keys_to_action)
    render_wrapper = env
    env = gym.wrappers.AtariPreprocessing(
        env, noop_max=0, frame_skip=config['env_frame_skip'], screen_size=config['env_frame_size'],
        terminal_on_life_loss=False, grayscale_obs=config['env_grayscale'])
    env = gym.wrappers.FrameStack(env, config['env_frame_stack'])
    if env.get_action_meanings()[1] == 'FIRE':
        env = utils.FireAfterLifeLoss(env)

    if user_input:
        while not render_wrapper.quit:
            env.reset(seed=seed)
            while True:
                a = render_wrapper.get_pressed_action()
                o, r, terminated, truncated, info = env.step(a)
                if terminated or truncated or render_wrapper.quit:
                    break
                render_wrapper.wait()
    else:
        agent = Agent(config, num_actions)
        agent.to(device)
        agent.load_state_dict(state_dict)
        agent.eval()

        while not render_wrapper.quit:
            dreamer = Dreamer(config, agent.wm, mode='observe', ac=agent.ac, store_data=False)
            o, _ = env.reset(seed=seed)
            o = utils.preprocess_atari_obs(o, device).unsqueeze(0).unsqueeze(1)
            dreamer.observe_reset_single(o)

            while True:
                a = dreamer.act(epsilon=0)
                o, r, terminated, truncated, info = env.step(a.item())
                o = utils.preprocess_atari_obs(o, device).unsqueeze(0).unsqueeze(1)
                r = torch.as_tensor([[r]], dtype=torch.float, device=device)
                terminated = torch.as_tensor([[terminated]], device=device)
                truncated = torch.as_tensor([[terminated]], device=device)
                dreamer.observe_step(a, o, r, terminated, truncated)
                if terminated or truncated or render_wrapper.quit:
                    break
                render_wrapper.wait()
    env.close()


def play_dream(config, state_dict, device, user_input, render_frame_stack, render_extra, seed):
    tmp_env = gym.make(f'{config["game"]}NoFrameskip-v4')
    tmp_env = tmp_env.unwrapped
    num_actions = tmp_env.action_space.n
    keys_to_action = tmp_env.get_keys_to_action()
    tmp_env.close()
    del tmp_env

    agent = Agent(config, num_actions)
    agent.to(device)
    agent.load_state_dict(state_dict)
    agent.eval()

    temperature = 1  # TODO
    if user_input:
        env = DreamEnv(
            config, agent.wm, temperature=temperature,
            render_mode='rgb_array', render_frame_stack=render_frame_stack, render_extra=render_extra)
    else:
        env = DreamEnv(
            config, agent.wm, temperature=temperature, ac=agent.ac,
            render_mode='rgb_array', render_frame_stack=render_frame_stack, render_extra=render_extra)

    zoom = 3
    fps = 60 / (config['env_frame_skip'] + 1)
    env = RenderWrapper(env, zoom, fps, keys_to_action)
    render_wrapper = env

    if user_input:
        while not render_wrapper.quit:
            env.reset(seed=seed)
            while True:
                a = render_wrapper.get_pressed_action()
                o, r, terminated, truncated, info = env.step(a)
                if terminated or truncated or render_wrapper.quit:
                    break
                render_wrapper.wait()
    else:
        while not render_wrapper.quit:
            env.reset(seed=seed)
            while True:
                a = env.act(epsilon=0)
                o, r, terminated, truncated, info = env.step(a.item())
                if terminated or truncated or render_wrapper.quit:
                    break
                render_wrapper.wait()
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--mode', type=str, default='dream')
    parser.add_argument('--input', type=str, default='user')
    parser.add_argument('--render_frame_stack', type=str, default='last')
    parser.add_argument('--render_extra', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    ii32 = np.iinfo(np.int32)
    seed = args.seed if args.seed is not None else np.random.randint(ii32.max)

    device = args.device
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']

    if args.mode in ('real', 'dream'):
        dream = args.mode == 'dream'
    else:
        print(f'Invalid mode: {args.mode}')
        exit(1)

    if args.input in ('agent', 'user'):
        user_input = args.input == 'user'
    else:
        print(f'Invalid input: {args.input}')
        exit(1)

    if dream:
        play_dream(
            config, state_dict, device, user_input, args.render_frame_stack, args.render_extra, seed)
    else:
        play_real(config, state_dict, device, user_input, seed)


if __name__ == '__main__':
    main()
