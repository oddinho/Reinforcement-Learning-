from gymnasium.envs.registration import register, registry

for env_id in ("cookiedisaster", "cookiedisaster/GridWorld-v0"):
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="cookiedisaster.envs:CookieDisasterEnv",
        )
