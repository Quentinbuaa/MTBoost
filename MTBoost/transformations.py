# transformations.py
from __future__ import annotations
import random
from typing import List
import torchvision.transforms.functional as TF


class TransformationParams:
    """
    Centralized definition for the 6-parameter MR transform:
    [rotation, translate_x, translate_y, scale, shear_x, shear_y]
    """
    RANGES = {
    "rotation": (-30.0, 30.0),
    "translate_x": (-3.0, 3.0),
    "translate_y": (-3.0, 3.0),
    "scale": (0.8, 1.2),
    "shear_x": (-10.0, 10.0),
    "shear_y": (-10.0, 10.0),
    }


    def __init__(self, values: List[float] | None = None):
        if values is None:
            values = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.rotation, self.tx, self.ty, self.scale, self.sx, self.sy = values
        self.clamp()


    # -------- Sampling / conversions -------- #
    @classmethod
    def sample_random(cls) -> "TransformationParams":
        return cls([
        round(random.uniform(*cls.RANGES["rotation"]), 2),
        round(random.uniform(*cls.RANGES["translate_x"]), 2),
        round(random.uniform(*cls.RANGES["translate_y"]), 2),
        round(random.uniform(*cls.RANGES["scale"]), 2),
        round(random.uniform(*cls.RANGES["shear_x"]), 2),
        round(random.uniform(*cls.RANGES["shear_y"]), 2),
        ])


    @classmethod
    def from_action(cls, action_6d: List[float]) -> "TransformationParams":
        """
        Map SAC action in [-1,1]^6 → valid param ranges (same mapping everywhere).
        """
        rotation = round(30.0 * action_6d[0], 2)
        tx = round(3.0 * action_6d[1], 2)
        ty = round(3.0 * action_6d[2], 2)
        scale = round(0.2 * action_6d[3] + 1.0, 2)
        sx = round(10.0 * action_6d[4], 2)
        sy = round(10.0 * action_6d[5], 2)
        return cls([rotation, tx, ty, scale, sx, sy])


    def to_list(self) -> List[float]:
        return [self.rotation, self.tx, self.ty, self.scale, self.sx, self.sy]


    # -------- Safety / clamping -------- #
    def clamp(self) -> None:
        self.rotation = _clamp(self.rotation, *self.RANGES["rotation"])
        self.tx = _clamp(self.tx, *self.RANGES["translate_x"])
        self.ty = _clamp(self.ty, *self.RANGES["translate_y"])
        self.scale = _clamp(self.scale, *self.RANGES["scale"])
        self.sx = _clamp(self.sx, *self.RANGES["shear_x"])
        self.sy = _clamp(self.sy, *self.RANGES["shear_y"])


    # -------- Application -------- #
    def apply(self, image):
        return TF.affine(
            image,
            angle=self.rotation,
            translate=[self.tx, self.ty],
            scale=self.scale,
            shear=[self.sx, self.sy],
            )


# ---------- helpers ---------- #
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x