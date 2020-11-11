import copy
from itertools import count

import numpy as np
import tensorflow as tf

from kaggle_environments import make
import kaggle_environments.envs.halite.helpers as hh

from gym_halite.envs.halite_env import get_scalar_features, get_feature_maps

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
            'kwargs': {
                'is_action_continuous': True
            },
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
            'max_path_length': 399,  # sample until terminal step or 'max_path_lenght',
            # then record it into the replay pool
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
            'n_epochs': 30,
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
    # it will iterate through the number of epochs 'n_epochs'
    # during epoch:
    # it will sample 'epoch_length' number of times (reset is not counted) to the pool
    # also, it will train each step, if there are more samples than 'min_pool_size' in the replay pool
    for i in count():
        if train_generator is None:
            train_generator = algorithm.train()
        diagnostics = next(train_generator)

        # it should be before printing to prevent a double print the last epoch
        try:
            if diagnostics['done']:
                break
        except KeyError:
            pass

        evalu_reward = diagnostics["evaluation"]["episode-reward-mean"]
        print(f"Evaluation: reward mean is {evalu_reward}")
        # train_reward = diagnostics["training"]["episode-reward-mean"]
        # print(f"Training: reward mean is {train_reward}")

    print("Finish")
    return policy


def digitize_action(action):
    if action < -0.6:
        action_number = 0  # north
    elif -0.6 <= action < -0.2:
        action_number = 1  # south
    elif -0.2 <= action < 0.2:
        action_number = 2  # west
    elif 0.2 <= action < 0.6:
        action_number = 3  # east
    elif 0.6 <= action:
        action_number = 4  # stay
    return action_number


def get_halite_agent(policy):
    """halite agent adapted for tf-agents policy"""
    def sac_halite_agent(state, config):
        from collections import OrderedDict

        directions = [hh.ShipAction.NORTH,
                      hh.ShipAction.SOUTH,
                      hh.ShipAction.WEST,
                      hh.ShipAction.EAST]

        board = hh.Board(state, config)
        me = board.current_player

        skalar_features = get_scalar_features(board)
        # skalar_features = skalar_features[np.newaxis, ...]
        # skalar_features = tf.convert_to_tensor(skalar_features, dtype=tf.float32)
        feature_maps = get_feature_maps(board)
        # feature_maps = feature_maps[np.newaxis, ...]
        # feature_maps = tf.convert_to_tensor(feature_maps, dtype=tf.float32)
        observation = OrderedDict({'feature_maps': feature_maps, 'scalar_features': skalar_features})

        action_step = policy.action(observation).numpy()
        action_number = digitize_action(action_step)
        # action_number = action_step.action.numpy()[0]
        try:
            me.ships[0].next_action = directions[action_number]
        except IndexError:
            pass
        return me.next_actions
    return sac_halite_agent


def render_halite(policy):
    board_size = 5
    starting_halite = 5000
    env = make("halite",
               configuration={"size": board_size,
                              "startingHalite": starting_halite},
               debug=True)

    halite_agent = get_halite_agent(policy)
    env.run([halite_agent])
    env.render(mode="ipython", width=800, height=600)


if __name__ == '__main__':
    policy = main(CONFIG)
