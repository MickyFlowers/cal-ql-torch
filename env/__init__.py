from gym.envs.registration import register

register(
    id="ur_env_v0",
    entry_point="env.env:UrEnv"
)