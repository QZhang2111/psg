import numpy as np
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as openpi_config

from psg.policy import PsgPolicy

config = openpi_config.get_config("pi05_libero")
checkpoint = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
policy = PsgPolicy(policy_config.create_trained_policy(config, checkpoint), config)  # a = 5, b = 1

request = {
    "observation/image": np.zeros((224, 224, 3), np.uint8),  # main camera
    "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
    "observation/state": np.zeros(8),  # end-effector position, axis-angle orientation, gripper joints
    "prompt": "put the bowl on the plate",
}
chunk = policy.infer(request)["actions"]
print(chunk.shape)  # (10, 7)
