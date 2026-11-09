"""Headless UI test: the Streamlit app must fool the vanilla CIFAR-10 model at ε = 8/255."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "checkpoints" / "cifar_cnn_vanilla.pt"
DATA = ROOT / "data" / "cifar10_test.pt"


@pytest.mark.skipif(not (CK.exists() and DATA.exists()), reason="needs trained checkpoint + cached CIFAR-10")
def test_ui_fools_vanilla_cifar_model_at_8_255():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(str(ROOT / "app" / "ui.py"), default_timeout=120)
    at.run()
    assert not at.exception
    at.selectbox[0].set_value("cifar_cnn_vanilla")
    at.selectbox[1].set_value("pgd_linf")
    at.slider[0].set_value(8)
    at.run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["L∞ / L2"].startswith("8.0/255")
    assert metrics["result"].startswith("FOOLED")
    assert len(at.get("image")) >= 1  # original/adversarial pair + triplet + confidence bars rendered
