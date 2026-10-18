"""U-Net: shapes, input channels, the loss, and whether it can learn at all."""

import numpy as np
import torch

from craquelure_fusion.generate_hard import HardConfig, generate_hard_sample
from craquelure_fusion.train_unet import TRAIN_SEED_START, TrainConfig, train
from craquelure_fusion.unet import UNet, dice_bce_loss, make_input, predict_scores
from craquelure_fusion.eval_set import SEED_START


def test_output_shape_matches_input() -> None:
    for in_ch in (3, 4):
        net = UNet(in_channels=in_ch, base=4, depth=3)
        out = net(torch.zeros(2, in_ch, 64, 64))
        assert out.shape == (2, 1, 64, 64)


def test_make_input_channels() -> None:
    s = generate_hard_sample(HardConfig(size=64, seed=1))
    assert make_input(s, use_height=False).shape == (3, 64, 64)
    assert make_input(s, use_height=True).shape == (4, 64, 64)


def test_predict_scores_in_unit_range_and_same_size() -> None:
    s = generate_hard_sample(HardConfig(size=64, seed=2))
    score = predict_scores(UNet(in_channels=4, base=4, depth=3), s, use_height=True)
    assert score.shape == (64, 64)
    assert score.min() >= 0 and score.max() <= 1


def test_dice_bce_is_small_when_perfect_and_large_when_wrong() -> None:
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4] = 1
    perfect = torch.where(target > 0, 10.0, -10.0)
    assert dice_bce_loss(perfect, target) < 0.01
    assert dice_bce_loss(-perfect, target) > 1.0


def test_training_seeds_do_not_overlap_eval_or_validation() -> None:
    eval_last = max(SEED_START.values()) + 1000  # room for any set size used so far
    assert TRAIN_SEED_START > eval_last


def test_can_overfit_one_small_batch() -> None:
    # A model that cannot memorise 4 crops has a bug (wrong shapes, detached loss, no grads).
    torch.manual_seed(0)
    samples = [generate_hard_sample(HardConfig(size=64, seed=s)) for s in (20000, 20001, 20002, 20003)]
    x = torch.from_numpy(np.stack([make_input(s, True) for s in samples]))
    y = torch.from_numpy(np.stack([s.crack_mask for s in samples]).astype(np.float32))[:, None]
    net = UNet(in_channels=4, base=8, depth=3)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    first = last = None
    for i in range(200):
        loss = dice_bce_loss(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if i == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    assert last < 0.3 * first, f"loss did not fall enough: {first:.3f} -> {last:.3f}"


def test_train_runs_and_saves_checkpoint(tmp_path) -> None:
    cfg = TrainConfig(use_height=False, steps=2, batch=2, crop=64, base=4)
    out = tmp_path / "ck.pt"
    losses = train(cfg, out)
    assert len(losses) == 2 and all(np.isfinite(losses))
    ck = torch.load(out, weights_only=False)
    assert ck["train_seeds"][0] == TRAIN_SEED_START


def test_saved_checkpoint_loads_as_scorer(tmp_path) -> None:
    from craquelure_fusion.learned import load_scorer

    out = tmp_path / "ck.pt"
    train(TrainConfig(use_height=True, steps=1, batch=2, crop=64, base=4), out)
    scorer = load_scorer(out)
    s = generate_hard_sample(HardConfig(size=64, seed=5))
    score = scorer(s)
    assert score.shape == (64, 64) and score.min() >= 0 and score.max() <= 1


def test_training_writes_progress_file_the_page_reads(tmp_path) -> None:
    import json

    out = tmp_path / "unet_x.pt"
    train(TrainConfig(use_height=False, steps=3, batch=2, crop=64, base=4), out)
    progress = json.loads((tmp_path / "unet_x.progress.json").read_text())
    assert progress["state"] == "done" and progress["step"] == progress["steps"] == 3
    assert progress["seeds_used"] == 6 and np.isfinite(progress["loss"])
