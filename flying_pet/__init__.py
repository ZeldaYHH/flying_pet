from gymnasium.envs.registration import register


register(
    id="FlyingPet-v0",
    entry_point="flying_pet.envs:FlyingPetEnv",
)
