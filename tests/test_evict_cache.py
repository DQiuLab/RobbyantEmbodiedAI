from src import evict_cache


def test_large_files_filters_by_size_and_walks_dirs(tmp_path):
    (tmp_path / "a").mkdir()
    big = tmp_path / "a" / "big.bin"; big.write_bytes(b"x" * 2048)
    small = tmp_path / "small.bin"; small.write_bytes(b"x" * 10)
    found = set(evict_cache.large_files([tmp_path], min_bytes=1024))
    assert found == {big}
    assert set(evict_cache.large_files([big], min_bytes=1024)) == {big}


def test_evict_runs_on_real_file_and_tolerates_missing(tmp_path):
    f = tmp_path / "f.bin"; f.write_bytes(b"y" * 4096)
    assert evict_cache.evict(f) is True
    assert evict_cache.evict(tmp_path / "missing") is False
    assert evict_cache.main([str(tmp_path), "--min-mb", "0"]) == 0
