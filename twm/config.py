CONFIGS = {}

CONFIGS['default'] = {
    # buffer
    'buffer_capacity': 100000,
    'buffer_temperature': 20.0,
    'buffer_prefill': 5000,

    # training
    'budget': 1000000000,
    'pretrain_budget': 50000000,
    'pretrain_obs_p': 0.6,
    'pretrain_dyn_p': 0.3,

    # evaluation
    'eval_every': 5000,
    'eval_episodes': 10,
    'final_eval_episodes': 100,

    # environment
    'env_frame_size': 64,
    'env_frame_skip': 4,
    'env_frame_stack': 4,
    'env_grayscale': True,
    'env_noop_max': 30,
    'env_time_limit': 27000,
    'env_episodic_lives': True,
    'env_reward_transform': 'tanh',
    'env_discount_factor': 0.99,
    'env_discount_lambda': 0.95,

    # world model
    'wm_batch_size': 100,
    'wm_sequence_length': 16,
    'wm_train_steps': 1,
    'wm_memory_length': 16,
    'wm_discount_threshold': 0.1,

    'z_categoricals': 32,
    'z_categories': 32,
    'obs_channels': 48,
    'obs_act': 'silu',
    'obs_norm': 'none',
    'obs_dropout': 0,
    'obs_lr': 1e-4,
    'obs_wd': 1e-6,
    'obs_eps': 1e-5,
    'obs_grad_clip': 100,
    'obs_entropy_coef': 5,
    'obs_entropy_threshold': 0.1,
    'obs_consistency_coef': 0.01,
    'obs_decoder_coef': 1,

    'dyn_embed_dim': 256,
    'dyn_num_heads': 4,
    'dyn_num_layers': 10,
    'dyn_feedforward_dim': 1024,
    'dyn_head_dim': 64,
    'dyn_z_dims': [512, 512, 512, 512],
    'dyn_reward_dims': [256, 256, 256, 256],
    'dyn_discount_dims': [256, 256, 256, 256],
    'dyn_input_rewards': True,
    'dyn_input_discounts': False,
    'dyn_act': 'silu',
    'dyn_norm': 'none',
    'dyn_dropout': 0.1,
    'dyn_lr': 1e-4,
    'dyn_wd': 1e-6,
    'dyn_eps': 1e-5,
    'dyn_grad_clip': 100,
    'dyn_z_coef': 1,
    'dyn_reward_coef': 10,
    'dyn_discount_coef': 50,

    # actor-critic
    'ac_batch_size': 400,
    'ac_horizon': 15,
    'ac_act': 'silu',
    'ac_norm': 'none',
    'ac_dropout': 0,
    'ac_input_h': False,
    'ac_h_norm': 'none',
    'ac_normalize_advantages': False,

    'actor_dims': [512, 512, 512, 512],
    'actor_lr': 1e-4,
    'actor_eps': 1e-5,
    'actor_wd': 1e-6,
    'actor_entropy_coef': 1e-2,
    'actor_entropy_threshold': 0.1,
    'actor_grad_clip': 1,

    'critic_dims': [512, 512, 512, 512],
    'critic_lr': 1e-5,
    'critic_eps': 1e-5,
    'critic_wd': 1e-6,
    'critic_grad_clip': 1,
    'critic_target_interval': 1
}
