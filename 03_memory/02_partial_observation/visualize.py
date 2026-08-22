from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from partial_dataset import PartialObservationSequenceDataset


ROOT = Path(__file__).resolve().parent


def _image(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    return np.clip(np.transpose(tensor, (1, 2, 0)), 0.0, 1.0)


def generate_visualizations(output_dir: Path = ROOT / "outputs") -> dict[str, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = PartialObservationSequenceDataset(num_sequences=2, sequence_length=6, seed=17)
    first, second = dataset[0], dataset[1]

    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(_image(first["full_worlds"][0]))
    axis.set_title("True full world at t=0\n(agent=(2,2), goal=(2,3))")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "full_world.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(_image(first["observations"][0]))
    axis.set_title("Agent-centred 3x3 observation at t=0\nouter cells = unknown")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "partial_observation.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(9, 6))
    for time_index in range(3):
        axes[0, time_index].imshow(_image(first["full_worlds"][time_index]))
        axes[0, time_index].set_title(f"true world t={time_index}")
        axes[1, time_index].imshow(_image(first["observations"][time_index]))
        visibility = "visible" if bool(first["goal_visible"][time_index]) else "outside view"
        axes[1, time_index].set_title(f"partial t={time_index}: goal {visibility}")
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle("Goal is seen at t=0 and outside the local view by t=2")
    figure.tight_layout()
    figure.savefig(output_dir / "observation_sequence.png", dpi=180)
    plt.close(figure)

    alias_time = dataset.alias_time
    identical = torch.equal(first["observations"][alias_time], second["observations"][alias_time])
    figure, axes = plt.subplots(2, 2, figsize=(6, 6))
    panels = (
        (first["observations"][0], "history A: t=0, goal right"),
        (first["observations"][alias_time], "A: t=2, goal outside"),
        (second["observations"][0], "history B: t=0, goal down"),
        (second["observations"][alias_time], "B: t=2, same observation"),
    )
    for axis, (image, title) in zip(axes.flat, panels, strict=True):
        axis.imshow(_image(image))
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"t=2 images are bitwise identical: {identical}")
    figure.tight_layout()
    figure.savefig(output_dir / "aliasing_pair.png", dpi=180)
    plt.close(figure)
    return {"alias_time_observations_identical": identical}


if __name__ == "__main__":
    print(generate_visualizations())
