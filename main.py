import os

from gemma import gm

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"


model = gm.nn.Gemma3_1B()

params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)
sampler = gm.text.ChatSampler(
    model=model,
    params=params,
    multi_turn=True,
)

turn0 = sampler.chat('Share one methapore linking "shadow" and "laughter".')
print(turn0)
