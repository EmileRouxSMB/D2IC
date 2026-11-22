from pathlib import Path

from d2ic.io.config import load_sequence_config


def test_alternating_dp600_config_uses_expected_roi_and_frame_limit() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_sequence_config(
        root / "img" / "DP600L1" / "PostWorkDir" / "alternating_workflow.toml"
    )
    assert config.image_stride == 10
    assert config.max_deformed_images == 24
    assert config.out_dir.name == "_outputs_stride10"
    assert (config.img_dir / config.mask_filename).resolve() == (
        root / "img" / "DP600L1" / "PostWorkDir" / "roi.tif"
    )
