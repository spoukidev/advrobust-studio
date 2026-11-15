import pickle
from pathlib import Path

import pytest
import torch

from studio.defenses.adv_train import save_checkpoint, train
from studio.models import TRUST_ENV_VAR, build_cifar_model, load_cifar_checkpoint, load_model

CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"


def test_models_forward():
    for arch in ("cifar_cnn", "cifar_resnet"):
        assert build_cifar_model(arch)(torch.rand(2, 3, 32, 32)).shape == (2, 10)


def test_checkpoint_roundtrip_and_tiny_training(tmp_path):
    x = (torch.rand(64, 3, 32, 32) * 255).to(torch.uint8)
    y = torch.randint(0, 10, (64,))
    for mode in ("vanilla", "pgd"):
        m, hist = train(x, y, arch="cifar_resnet", mode=mode, epochs=1, batch_size=32, train_steps=1, log=lambda *_: None)
        assert len(hist) == 1
        p = tmp_path / f"{mode}.pt"
        save_checkpoint(m, p, "cifar_resnet")
        m2 = load_model(str(p))
        xx = torch.rand(2, 3, 32, 32)
        assert torch.allclose(m(xx), m2(xx), atol=1e-5)
    # raw state_dict also supported (and safe)
    torch.save(m.state_dict(), tmp_path / "raw.pt")
    m3 = load_cifar_checkpoint(tmp_path / "raw.pt", arch="cifar_resnet")
    assert torch.allclose(m(xx), m3(xx), atol=1e-5)


def test_saved_checkpoint_is_weights_only_loadable(tmp_path):
    p = tmp_path / "ck.pt"
    save_checkpoint(build_cifar_model("cifar_cnn"), p, "cifar_cnn", meta={"mode": "vanilla", "epochs": 1})
    ck = torch.load(p, weights_only=True)
    assert ck["arch"] == "cifar_cnn" and ck["meta"]["epochs"] == 1


@pytest.mark.parametrize("name", ["cifar_cnn_vanilla", "cifar_cnn_pgdat", "cifar_resnet_vanilla"])
def test_committed_checkpoints_load_safely(name):
    p = CHECKPOINTS / f"{name}.pt"
    if not p.exists():
        pytest.skip("checkpoint not present")
    assert load_cifar_checkpoint(p)(torch.rand(1, 3, 32, 32)).shape == (1, 10)


def test_legacy_full_module_pickle_requires_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv(TRUST_ENV_VAR, raising=False)
    p = tmp_path / "full.pt"
    torch.save(build_cifar_model("cifar_cnn"), p)
    with pytest.raises(pickle.UnpicklingError, match=TRUST_ENV_VAR):
        load_cifar_checkpoint(p)
    with pytest.warns(UserWarning, match="weights_only=False"):
        assert load_cifar_checkpoint(p, trust_checkpoint=True)(torch.rand(1, 3, 32, 32)).shape == (1, 10)
    monkeypatch.setenv(TRUST_ENV_VAR, "1")
    with pytest.warns(UserWarning):
        assert load_model(str(p))(torch.rand(1, 3, 32, 32)).shape == (1, 10)
    # an explicit False beats the env var
    with pytest.raises(pickle.UnpicklingError):
        load_cifar_checkpoint(p, trust_checkpoint=False)


class _Payload:
    """Pickle gadget: unpickling it would create a marker file."""

    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (Path.touch, (Path(self.marker),))


def test_malicious_pickle_is_rejected_without_running_code(tmp_path, monkeypatch):
    monkeypatch.delenv(TRUST_ENV_VAR, raising=False)
    marker = tmp_path / "pwned"
    p = tmp_path / "evil.pt"
    torch.save({"arch": "cifar_cnn", "state_dict": {}, "meta": _Payload(marker)}, p)
    with pytest.raises(pickle.UnpicklingError):
        load_cifar_checkpoint(p)
    with pytest.raises(pickle.UnpicklingError):
        load_model(str(p))
    assert not marker.exists()
