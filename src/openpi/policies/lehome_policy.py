"""LeHome Challenge policy transforms.

Maps between the LeHome simulation observation format and the openpi model's
expected input/output format.

Dataset feature keys (dot-separated, from LeRobot format):
    observation.state          (12,)  — dual-arm joint angles
    observation.images.top_rgb (H,W,3) uint8 — overhead camera
    observation.images.left_rgb  (H,W,3) uint8 — left wrist camera
    observation.images.right_rgb (H,W,3) uint8 — right wrist camera
    action                     (12,)  — absolute joint targets

After the repack transform (see lehome_config.py), data arrives here as:
    images.cam_high       — top camera
    images.cam_left_wrist — left camera
    images.cam_right_wrist — right camera
    state                 — (12,) joint angles
    actions               — (action_horizon, 12) joint targets
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Action dimension for LeHome dual-arm SO-ARM101 (6 joints × 2 arms).
ACTION_DIM = 12
STATE_DIM = 12

# Default language prompt used when the dataset has no task field.
DEFAULT_PROMPT = "fold and place the garment on the bed"


def make_lehome_example() -> dict:
    """Creates a random input example matching the LeHome inference observation format.

    Keys mirror what the lehome-challenge docker server receives from the simulator.
    """
    return {
        "observation/state": np.zeros(STATE_DIM, dtype=np.float32),
        "observation/images/top_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation/images/left_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation/images/right_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "prompt": DEFAULT_PROMPT,
    }


def _parse_image(image) -> np.ndarray:
    """Ensure image is uint8 (H, W, C).

    LeRobot stores images as float32 (C, H, W); during policy inference they
    arrive as uint8 (H, W, C) from the simulator.  This handles both.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LeHomeInputs(transforms.DataTransformFn):
    """Converts LeHome observations into the openpi model input format.

    Used for both training (data comes from the LeRobot dataset after repack)
    and inference (data comes from the Docker HTTP server).

    Expected keys after repack (training) or direct (inference):
        images.cam_high        — top overhead camera
        images.cam_left_wrist  — left wrist camera
        images.cam_right_wrist — right wrist camera
        state                  — (12,) joint angles
        actions                — (action_horizon, 12), training only
        prompt                 — str, optional
    """

    model_type: _model.ModelType

    # Camera names present in the LeHome dataset.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(
                f"Unexpected cameras {unexpected}. Expected: {self.EXPECTED_CAMERAS}"
            )

        base_image = _parse_image(in_images["cam_high"])

        images: dict[str, np.ndarray] = {"base_0_rgb": base_image}
        image_masks: dict[str, np.ndarray] = {"base_0_rgb": np.True_}

        for dest, src in (
            ("left_wrist_0_rgb", "cam_left_wrist"),
            ("right_wrist_0_rgb", "cam_right_wrist"),
        ):
            if src in in_images:
                images[dest] = _parse_image(in_images[src])
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs: dict = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"], dtype=np.float32),
        }

        # Actions only available during training.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LeHomeOutputs(transforms.DataTransformFn):
    """Converts model output back to LeHome joint-space actions.

    The model pads states/actions to its internal action_dim.  We strip
    the padding here and return the first ACTION_DIM (12) columns.
    """

    def __call__(self, data: dict) -> dict:
        # data["actions"] shape: (action_horizon, model_action_dim)
        # Slice to the actual robot action dimension (12).
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
