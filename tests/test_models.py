import torch

from studio.defenses.adv_train import save_checkpoint, train
from studio.models import build_cifar_model, load_cifar_checkpoint, load_model


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
    # raw module pickle also supported
    torch.save(m, tmp_path / "full.pt")
    assert load_cifar_checkpoint(tmp_path / "full.pt")(torch.rand(1, 3, 32, 32)).shape == (1, 10)
