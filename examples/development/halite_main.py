import copy
import tensorflow as tf

from softlearning import value_functions
from softlearning import policies
from softlearning import replay_pools
from softlearning import samplers
from softlearning import algorithms
from softlearning.environments.utils import get_environment_from_params

CONFIG = {
    'environment_params': {
        'training': {
            'domain': 'gym_halite:halite',
            'kwargs': {},
            'task': 'v0',
            'universe': 'gym',
        },
    },
    'Q_params': {
        'class_name': 'double_halite_Q_function',
        'config': {
            'observation_keys': None,
            'preprocessors': None,
        },
    },
    'policy_params': {
        'class_name': 'HaliteGaussianPolicy',
        'config': {
            'observation_keys': None,
            'preprocessors': None,
            'squash': True,
            'dtypes': tf.float32,
        },
    },
    'replay_pool_params': {
        'class_name': 'SimpleReplayPool',
        'config': {
            'max_size': 39900  # the total number of steps (samples)
        },
    },
    'sampler_params': {
        'class_name': 'SimpleSampler',
        'config': {
            'max_path_length': 399,  # the total number of steps in epizode
        },
    },
    'algorithm_params': {
        'class_name': 'SAC',
        'config': {
            'Q_lr': 0.0003,
            'alpha_lr': 0.0003,
            'batch_size': 4,
            'discount': 0.99,
            'epoch_length': 399,  # train generator yields after an epoch
            'eval_n_episodes': 1,
            'eval_render_kwargs': {},
            'min_pool_size': 399,  # ready_to_train = min_pool_size <= pool.size (a number of samples)
            'n_epochs': 301,
            'n_train_repeat': 1,
            'num_warmup_samples': 0,
            'policy_lr': 0.0003,
            'reward_scale': 1.0,
            'target_entropy': 'auto',
            'target_update_interval': 1,
            'tau': 0.005,
            'train_every_n_steps': 1,
            'video_save_frequency': 0,
        },
    },
}


def main(variant_in):
    variant = copy.deepcopy(variant_in)

    environment_params = variant['environment_params']
    training_environment = get_environment_from_params(environment_params['training'])
    evaluation_environment = (
        get_environment_from_params(environment_params['evaluation'])
        if 'evaluation' in environment_params else training_environment
    )

    variant['Q_params']['config'].update({
        'input_shapes': (
            training_environment.observation_shape,
            training_environment.action_shape),
    })
    Qs = value_functions.get(variant['Q_params'])

    variant['policy_params']['config'].update({
        'action_range': (training_environment.action_space.low,
                         training_environment.action_space.high),
        'input_shapes': training_environment.observation_shape,
        'output_shape': training_environment.action_shape,
    })
    policy = policies.get(variant['policy_params'])

    variant['replay_pool_params']['config'].update({
        'environment': training_environment,
    })
    replay_pool = replay_pools.get(variant['replay_pool_params'])

    variant['sampler_params']['config'].update({
        'environment': training_environment,
        'policy': policy,
        'pool': replay_pool,
    })
    sampler = samplers.get(variant['sampler_params'])

    variant['algorithm_params']['config'].update({
        'training_environment': training_environment,
        'evaluation_environment': evaluation_environment,
        'policy': policy,
        'Qs': Qs,
        'pool': replay_pool,
        'sampler': sampler
    })
    algorithm = algorithms.get(variant['algorithm_params'])
    print("Initialization finished")

    train_generator = None
    for i in range(10):
        if train_generator is None:
            train_generator = algorithm.train()
        diagnostics = next(train_generator)
        evalu_reward = diagnostics["evaluation"]["episode-reward-mean"]
        train_reward = diagnostics["training"]["episode-reward-mean"]
        print(f"Evaluation: reward mean is {evalu_reward}")
        print(f"Training: reward mean is {train_reward}")

    print("Finish")


if __name__ == '__main__':
    main(CONFIG)
