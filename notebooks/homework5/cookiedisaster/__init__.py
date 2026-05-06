from gymnasium.envs.registration import register

register(
    id='cookiedisaster/GridWorld-v0',
    entry_point="cookiedisaster.envs:CookieDisasterEnv",
)