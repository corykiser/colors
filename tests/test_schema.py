import pytest
from palette.schema import PaletteRecord, records_to_frame, frame_to_records, write_parquet, read_parquet


def mk(**kw):
    base = dict(record_id="x", source_family="test", source_release="v0", source_url="u",
                original_color_space="sRGB", original_colors=[[0, 0, 0], [1, 1, 1]],
                oklab_colors=[[0, 0, 0], [1, 0, 0]], label_type="curated")
    base.update(kw)
    return PaletteRecord(**base)


def test_validation():
    r = mk()
    assert r.original_order == [0, 1] and r.n_colors == 2
    with pytest.raises(ValueError):
        mk(label_type="likes")
    with pytest.raises(ValueError):
        mk(oklab_colors=[[0, 0, 0]])
    with pytest.raises(ValueError):
        mk(original_order=[1, 1])
    with pytest.raises(ValueError):
        mk(rating_count=3, individual_ratings=[{"r": 1}])


def test_parquet_roundtrip(tmp_path):
    rs = [mk(record_id="a", rating_count=2, individual_ratings=[{"user": 1, "rating": 3}, {"user": 2, "rating": 5}],
             mean_rating=4.0, quality_flags=["x"]),
          mk(record_id="b", text_description="sunset")]
    p = tmp_path / "t.parquet"
    write_parquet(rs, str(p))
    back = read_parquet(str(p))
    assert back == rs
