import random
from typing import Set

import numpy as np


def random_rotate(x: np.ndarray,
                  rotation_directions: Set[int],
                  mirror_directions: Set[int] = None,
                  mirror_first=True) -> np.ndarray:
    if mirror_directions is None:
        mirror_directions = rotation_directions

    rotation_directions = list(rotation_directions)
    mirror_directions = list(mirror_directions)

    def _random_rotate(x_local: np.ndarray) -> np.ndarray:
        original_directions = rotation_directions.copy()
        random.shuffle(rotation_directions)
        for rotate_from, rotate_to in zip(original_directions, rotation_directions):
            if rotate_from == rotate_to:
                continue
            x_local = np.rot90(x_local, k=1, axes=(rotate_from, rotate_to))
        return x_local

    if mirror_first:
        for mirror_direction in mirror_directions:
            if random.random() < 0.5:
                x = np.flip(x, axis=mirror_direction)
        x = _random_rotate(x)
    else:
        x = _random_rotate(x)
        for mirror_direction in mirror_directions:
            if random.random() < 0.5:
                x = np.flip(x, axis=mirror_direction)
    return x
