"""Mask paths may live outside the visible-image directory."""

from pathlib import Path

import pytest

from d2ic.io.config import load_sequence_config


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/configs/sequential_dic.toml"


def test_explicit_mask_path_is_relative_to_config_and_overrides_legacy_name(tmp_path):
    cfg_path = tmp_path / "config.toml"
    source = EXAMPLE.read_text(encoding="utf-8")
    source = source.replace(
        'mask_filename = "roi.tif"',
        'mask_filename = "old_roi.tif"\nmask_path = "roi/mask_final.tif"',
    )
    cfg_path.write_text(source, encoding="utf-8")

    cfg = load_sequence_config(cfg_path)

    assert cfg.mask_path == tmp_path / "roi/mask_final.tif"
    assert cfg.mask_filename == "old_roi.tif"
    assert cfg.mask_path != cfg.img_dir / cfg.mask_filename


def test_explicit_mask_path_does_not_require_legacy_filename(tmp_path):
    cfg_path = tmp_path / "config.toml"
    source = EXAMPLE.read_text(encoding="utf-8")
    source = source.replace('mask_filename = "roi.tif"', 'mask_path = "roi/mask_final.tif"')
    cfg_path.write_text(source, encoding="utf-8")

    cfg = load_sequence_config(cfg_path)

    assert cfg.mask_filename is None
    assert cfg.mask_path == tmp_path / "roi/mask_final.tif"


def test_absolute_mask_path_is_preserved(tmp_path):
    cfg_path = tmp_path / "config.toml"
    absolute_mask = tmp_path / "external" / "mask.tif"
    source = EXAMPLE.read_text(encoding="utf-8")
    cfg_path.write_text(
        source.replace('mask_filename = "roi.tif"', f'mask_path = "{absolute_mask}"'),
        encoding="utf-8",
    )

    assert load_sequence_config(cfg_path).mask_path == absolute_mask


def test_mask_path_or_filename_is_required(tmp_path):
    cfg_path = tmp_path / "config.toml"
    source = EXAMPLE.read_text(encoding="utf-8")
    cfg_path.write_text(source.replace('mask_filename = "roi.tif"\n', ''), encoding="utf-8")

    with pytest.raises(ValueError, match="mask_path or mask_filename"):
        load_sequence_config(cfg_path)
