"""PMT selection and distillation contracts, without policy training."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.config import TrainConfig
from supermimic.eval.benchmark import ClipResult
from supermimic.perception.control import aligned_action, pd_target_loss
from supermimic.perception.distillation import DistillConfig, EpisodeTeacherMix, distill
from supermimic.perception.filtering import (
    pmt_ik_acceptance,
    selected_reference_names,
    tracking_errors,
    tracking_failures,
    write_teacher_filter,
)
from supermimic.perception.paired import configure_stage, sha256, validate_student_initialization
from supermimic.perception.terrain import TerrainCfg


def test_official_ik_coarse_filter_counts_bad_frames_not_global_max():
    assert pmt_ik_acceptance([0.19] * 100)["accepted"]
    assert pmt_ik_acceptance([0.200] * 9)["accepted"]
    assert pmt_ik_acceptance([0.201] * 8 + [0.0])["accepted"]
    assert not pmt_ik_acceptance([0.201] * 9)["accepted"]
    with pytest.raises(ValueError, match="finite"):
        pmt_ik_acceptance([float("nan")])


def test_pmt_filter_detects_drift_and_end_effector_loss_despite_upright_root():
    ref = torch.zeros(1, 5, 3)
    quat = torch.zeros(1, 5, 4)
    quat[..., 0] = 1
    actual = ref.clone() + torch.tensor([1.0, 0.0, 0.0])
    errors = tracking_errors(actual, quat, ref, quat, 0, [1, 2, 3, 4])
    torch.testing.assert_close(errors, torch.tensor([[1.0, 0.0, 0.0]]))
    assert tracking_failures(errors).tolist() == [[True, False, False]]
    actual = ref.clone()
    actual[:, 3, 0] = 0.36
    assert tracking_failures(tracking_errors(actual, quat, ref, quat, 0, [1, 2, 3, 4])).tolist() == [[False, False, True]]
    rotated = quat.clone()
    rotated[:, 0, :2] = 2**-0.5
    assert tracking_failures(tracking_errors(ref, rotated, ref, quat, 0, [1, 2, 3, 4]))[0, 1]


def test_pmt_eval_does_not_reintroduce_legacy_height_termination():
    from supermimic.envs import SuperMimicEnvCfgArgs
    from supermimic.envs.tracking_env_cfg import make_env_cfg

    args = SuperMimicEnvCfgArgs(npz_files=[], num_envs=1, record_transitions=False,
                                terrain=TerrainCfg(enabled=True), pmt_tracking_evaluation=True)
    terms = make_env_cfg(args).terminations
    assert set(terms) == {"time_out", "clip_end", "pmt_tracking"}
    args.pmt_tracking_evaluation = False
    assert set(make_env_cfg(args).terminations) == {"time_out", "clip_end", "root_height", "root_tilt"}


@pytest.fixture
def pairs(tmp_path):
    directory = tmp_path / "pairs"
    manifest = {"format": "supermimic-tcrs-v1", "split": "train",
                "terrain": asdict(TerrainCfg(enabled=True)), "clips": []}
    for sub in ("raw", "adapted"):
        (directory / sub).mkdir(parents=True)
    for name in ("low_return_but_tracks", "high_return_but_fails"):
        record = {"name": name, "metrics": {"accepted": True, "frames": 10}}
        for sub in ("raw", "adapted"):
            p = directory / sub / f"{name}.npz"
            np.savez(p, joint_pos=np.zeros((10, 29)), joint_names=np.array([f"q{i}" for i in range(29)]), fps=50)
            record[f"{sub}_sha256"] = sha256(p)
        manifest["clips"].append(record)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"frozen teacher identity")
    results = [
        ClipResult("low_return_but_tracks", 10, 10, 0.1, False, -1,
                   tracking_max={"anchor_position_m": 0.1, "anchor_gravity_z": 0.1,
                                 "end_effector_position_m": 0.1}, failure_reasons=[]),
        ClipResult("high_return_but_fails", 10, 3, 0.9, True, 3,
                   tracking_max={"anchor_position_m": 0.36, "anchor_gravity_z": 0.0,
                                 "end_effector_position_m": 0.1}, failure_reasons=["anchor_position_m"]),
    ]
    return directory, teacher, results, tmp_path / "selection.json"


def test_filter_keeps_trackable_reference_without_return_cutoff_and_binds_teacher(pairs):
    directory, teacher, results, out = pairs
    receipt = write_teacher_filter(directory, teacher, results, out)
    assert receipt["kept_count"] == 1
    assert selected_reference_names(directory, out, teacher) == ["low_return_but_tracks"]
    teacher.write_bytes(b"different teacher")
    with pytest.raises(ValueError, match="different teacher"):
        selected_reference_names(directory, out, teacher)


def test_partial_survival_only_and_heldout_evaluations_cannot_certify_data(pairs):
    directory, teacher, results, out = pairs
    with pytest.raises(ValueError, match="every candidate"):
        write_teacher_filter(directory, teacher, results[:1], out)
    results[0].tracking_max = None
    with pytest.raises(ValueError, match="PMT tracking"):
        write_teacher_filter(directory, teacher, results, out)
    manifest = directory / "manifest.json"
    values = json.loads(manifest.read_text())
    values["split"] = "validation"
    manifest.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="validation/test"):
        write_teacher_filter(directory, teacher, results, out)


def test_formal_dagger_and_student_cannot_skip_selection(pairs, tmp_path):
    directory, teacher, _, _ = pairs
    with pytest.raises(ValueError, match="filter_teacher_references"):
        distill(DistillConfig(teacher=teacher, paired_dir=directory, output=tmp_path / "student.pt"))
    cfg = TrainConfig(stage="student")
    cfg.data.paired_dir = directory
    with pytest.raises(ValueError, match="teacher filter"):
        configure_stage(cfg)
    cfg.integration_test = True
    with pytest.raises(ValueError, match="integration-test bypass"):
        configure_stage(cfg)


def test_student_uses_only_retained_references_and_matching_distillation(pairs):
    directory, teacher, results, out = pairs
    write_teacher_filter(directory, teacher, results, out)
    cfg = TrainConfig(stage="student", init_policy="distilled.pt", init_world_model=teacher)
    cfg.data.paired_dir, cfg.data.teacher_filter = directory, out
    raw, adapted = configure_stage(cfg)
    assert len(raw) == len(adapted) == 1
    assert "low_return_but_tracks" in raw[0]
    checkpoint = {"stage": "distilled_student", "teacher_filter_sha256": sha256(out)}
    validate_student_initialization(cfg, checkpoint)
    with pytest.raises(ValueError, match="different teacher-filter"):
        validate_student_initialization(cfg, {**checkpoint, "teacher_filter_sha256": "bad"})
    with pytest.raises(ValueError, match="distilled-student"):
        validate_student_initialization(cfg, {**checkpoint, "stage": "teacher"})


def test_physical_pd_labels_and_loss_are_invariant_to_action_units():
    t = {"joint_names": ["a", "b"], "scale": [0.2, 0.4], "offset": [0.1, 0.5]}
    s = {"joint_names": ["a", "b"], "scale": [0.3, 0.2], "offset": [-0.1, 0.7]}
    teacher = torch.tensor([[0.5, -0.4]])
    target = aligned_action(teacher, t, s)
    prediction = torch.zeros_like(target, requires_grad=True)
    expected = ((torch.tensor(s["offset"]) + prediction * torch.tensor(s["scale"]))
                - (torch.tensor(t["offset"]) + teacher * torch.tensor(t["scale"]))).square().mean()
    loss = pd_target_loss(prediction, target, s)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert prediction.grad.abs().sum() > 0


def test_episode_mixing_never_switches_an_unfinished_episode():
    mix = EpisodeTeacherMix(3, 1.0, "cpu")
    t, s = torch.ones(3, 2), torch.zeros(3, 2)
    mix.reset(torch.tensor([False, False, False]), 0.0)
    torch.testing.assert_close(mix.apply(t, s), t)
    mix.reset(torch.tensor([False, True, False]), 0.0)
    torch.testing.assert_close(mix.apply(t, s), torch.tensor([[1., 1.], [0., 0.], [1., 1.]]))


def test_student_backbone_lr_scale_survives_scheduler_updates():
    from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
    from supermimic.policy.nets import Policy, PolicyCfg

    policy = Policy(PolicyCfg(encoder_hidden=(8,), decoder_hidden=(8,), terrain_dim=374))
    cfg = PolicyTrainerCfg(compile=None, backbone_lr_scale=0.3)
    trainer = PolicyTrainer(SimpleNamespace(terrain=None), policy, torch.nn.Linear(2, 1),
                            SimpleNamespace(default_joint_pos=torch.zeros(29)), cfg)
    groups = trainer.opt.param_groups
    assert len(groups) == 2
    assert sum(len(g["params"]) for g in groups) == len(list(policy.parameters()))
    trainer.set_lr(1e-4, 2e-4)
    assert [g["lr"] for g in groups] == pytest.approx([3e-5, 1e-4])
    assert trainer.critic_opt.param_groups[0]["lr"] == pytest.approx(2e-4)


def test_synthesis_records_failed_candidates_and_continues(tmp_path, monkeypatch):
    from scripts import synthesize_terrain

    source = tmp_path / "source"
    source.mkdir()
    for name, value in (("bad", -1), ("good", 0)):
        np.savez(source / f"{name}.npz", fps=50, joint_pos=np.full((3, 29), value),
                 joint_vel=np.zeros((3, 29)), body_pos_w=np.zeros((3, 1, 3)),
                 body_quat_w=np.zeros((3, 1, 4)), body_lin_vel_w=np.zeros((3, 1, 3)),
                 body_ang_vel_w=np.zeros((3, 1, 3)))

    class StubSynthesizer:
        def __init__(self, *args):
            pass

        def synthesize(self, raw):
            if raw["joint_pos"][0, 0] < 0:
                raise ValueError("no support contacts")
            return raw, {"frames": 3, "accepted": True}

    monkeypatch.setattr(synthesize_terrain, "Synthesizer", StubSynthesizer)
    out = tmp_path / "paired"
    synthesize_terrain.generate(synthesize_terrain.Config(npz_dir=source, output=out))
    report = json.loads((out / "manifest.json").read_text())
    assert report["candidate_count"] == 2 and report["accepted_count"] == 1
    assert report["clips"][0]["metrics"]["reason"] == "synthesis_failure"
    assert (out / "rejected/bad.npz").is_file()
    assert (out / "raw/good.npz").is_file()
    assert (source / "bad.npz").is_file()
