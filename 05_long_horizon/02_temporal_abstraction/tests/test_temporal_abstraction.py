import sys
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from macro_dataset import MacroSequenceDataset, chunk_sequences  # noqa: E402
from macro_dynamics import ActionChunkEncoder, MacroDynamics  # noqa: E402


def test_chunked_dataset_has_correct_boundaries_and_shapes() -> None:
    dataset = MacroSequenceDataset(8, sequence_length=30, chunk_size=5, seed=2)
    assert dataset.states.shape == (8, 31, 2)
    assert dataset.boundary_states.shape == (8, 7, 2)
    assert dataset.action_chunks.shape == (8, 6, 5, 4)
    assert torch.equal(dataset.boundary_states, dataset.states[:, ::5])
    assert torch.equal(dataset.action_chunks.reshape(8, 30, 4), dataset.actions)


def test_invalid_chunk_divisibility_is_rejected() -> None:
    states = torch.zeros(2, 8, 2)
    actions = torch.zeros(2, 7, 4)
    try:
        chunk_sequences(states, actions, chunk_size=5)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("expected chunk validation")


def test_action_chunk_encoder_preserves_leading_dimensions() -> None:
    encoder = ActionChunkEncoder()
    actions = torch.nn.functional.one_hot(torch.randint(0, 4, (3, 6, 5)), 4).float()
    embeddings = encoder(actions)
    assert embeddings.shape == (3, 6, 32)
    assert torch.isfinite(embeddings).all()


def test_macro_forward_rollout_shapes_and_gradients() -> None:
    model = MacroDynamics()
    dataset = MacroSequenceDataset(4, 30, 5, seed=4)
    teacher_forced = model(dataset.boundary_states[:, :-1], dataset.action_chunks)
    rollout = model.rollout(dataset.boundary_states[:, 0], dataset.action_chunks)
    assert teacher_forced.shape == rollout.shape == (4, 6, 2)
    torch.nn.functional.mse_loss(teacher_forced, dataset.boundary_states[:, 1:]).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_macro_rollout_does_not_consume_future_boundary_truth() -> None:
    model = MacroDynamics()
    dataset = MacroSequenceDataset(2, 30, 5, seed=5)
    expected = model.rollout(dataset.boundary_states[:, 0], dataset.action_chunks)
    corrupted = dataset.boundary_states.clone()
    corrupted[:, 1:] += 1000.0
    actual = model.rollout(corrupted[:, 0], dataset.action_chunks)
    assert torch.equal(expected, actual)
