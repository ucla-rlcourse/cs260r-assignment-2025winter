"""
This is an example of how to use the SafeMetaDriveEnv environment.
We will use the same VALIDATION_CONFIG below to evaluate the "baseline performance" of the trained agent.
A hidden test set will be used to evaluate the "final performance" of your trained agent.

You can run this file directly to use keyboard to control the vehicle in the training env.
"""
import copy

from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

DEFAULT_CONFIG = {
    # The below are default configs copied from SafeMetaDriveEnv
    # Environment difficulty
    "accident_prob": 0.8,
    "traffic_density": 0.05,
    # Termination conditions
    "crash_vehicle_done": False,
    "crash_object_done": False,
    # Reward
    "success_reward": 10.0,
    "driving_reward": 1.0,
    "speed_reward": 0.1,
    # Penalty will be negated and added to reward
    "out_of_road_penalty": 5.0,
    "crash_vehicle_penalty": 1.0,
    "crash_object_penalty": 1.0,
    # Cost will be return in info["cost"] and you can do constrained optimization with it
    "crash_vehicle_cost": 1.0,
    "crash_object_cost": 1.0,
    "out_of_road_cost": 1.0,
    "horizon": 1000,
}

# Use deepcopy to avoid modifying the DEFAULT_CONFIG
TRAINING_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
TRAINING_CONFIG.update(
    {  # Environment setting
        "num_scenarios": 50,  # There are totally 50 possible maps.
        "start_seed": 100,  # We will use the map with seeds in [100, 150) as the default training environment.
    }
)


def get_training_env(extra_config=None):
    config = copy.deepcopy(TRAINING_CONFIG)
    if extra_config:
        config.update(extra_config)
    return SafeMetaDriveEnv(config)


VALIDATION_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
VALIDATION_CONFIG.update(
    {  # Environment setting
        "num_scenarios": 50,  # There are totally 50 possible maps.
        "start_seed": 1000,  # We will use the map with seeds in [1000, 1050) as the default validation environment.
    }
)


def get_validation_env(extra_config=None):
    config = copy.deepcopy(VALIDATION_CONFIG)
    if extra_config:
        config.update(extra_config)
    return SafeMetaDriveEnv(config)


if __name__ == "__main__":
    env = get_training_env({
        "manual_control": True,
        "use_render": True,
    })
    env.reset()
    env.engine.toggle_help_message()  # Show the help message in the rendering window
    while True:
        _, _, tm, tc, _ = env.step([0, 0])
        env.render(mode="topdown", target_agent_heading_up=True)
        done = tm or tc
        if done:
            env.reset()
