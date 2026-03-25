from __future__ import annotations

from dataclasses import dataclass

from lerobot.transforms.core import DataDict, DataTransformFn
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@DataTransformFn.register_subclass("unify_f1_inputs")
@dataclass
class UnifyF1InputsTransformFn(DataTransformFn):
    """Keep only the fields required by the f1 policy."""

    def __call__(self, data: DataDict) -> DataDict:
        return {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            f"{OBS_IMAGES}.image0": data[f"{OBS_IMAGES}.image0"],
            f"{OBS_IMAGES}.image0_mask": data[f"{OBS_IMAGES}.image0_mask"],
            f"{OBS_IMAGES}.image0_is_pad": data[f"{OBS_IMAGES}.image0_is_pad"],
        }
