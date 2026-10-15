from src.config_loader import load_config


def test_load_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("timeout: 5\n")
    assert load_config(p)["timeout"] == 5
