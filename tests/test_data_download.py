"""Download layer tests. They run offline: fetch() is pointed at file:// URLs."""

import hashlib

import pytest

from qm9dipole.data import QM9_FILES, ChecksumError, RawFile, fetch


@pytest.fixture
def source(tmp_path):
    """A local 'remote' file and the RawFile record that correctly describes it."""
    payload = b"fake qm9 payload\n" * 1000
    src = tmp_path / "remote" / "blob.bin"
    src.parent.mkdir()
    src.write_bytes(payload)
    raw = RawFile("blob.bin", 0, len(payload), hashlib.md5(payload).hexdigest())
    return src, raw


def test_fresh_download_verifies_and_leaves_no_part_file(tmp_path, source):
    src, raw = source
    dest_dir = tmp_path / "raw"
    path = fetch(raw, dest_dir, url=src.as_uri())
    assert path.read_bytes() == src.read_bytes()
    assert not list(dest_dir.glob("*.part"))


def test_bad_checksum_raises_and_leaves_nothing(tmp_path, source):
    src, raw = source
    bad = RawFile(raw.name, 0, raw.size, "0" * 32)
    dest_dir = tmp_path / "raw"
    with pytest.raises(ChecksumError):
        fetch(bad, dest_dir, url=src.as_uri())
    assert not any(dest_dir.iterdir())


def test_rerun_with_intact_file_skips_download(tmp_path, source):
    src, raw = source
    dest_dir = tmp_path / "raw"
    first = fetch(raw, dest_dir, url=src.as_uri())
    # A second fetch must not touch the network: point it at a URL that cannot be opened.
    missing = (tmp_path / "does-not-exist.bin").as_uri()
    assert fetch(raw, dest_dir, url=missing) == first


def test_rerun_with_modified_file_raises_and_keeps_it(tmp_path, source):
    src, raw = source
    dest_dir = tmp_path / "raw"
    path = fetch(raw, dest_dir, url=src.as_uri())
    tampered = b"X" + path.read_bytes()[1:]  # same size, different content
    path.write_bytes(tampered)
    with pytest.raises(ChecksumError, match="corrupt or was modified"):
        fetch(raw, dest_dir, url=src.as_uri())
    assert path.read_bytes() == tampered  # left in place for a human to inspect


def test_manifest_matches_plan():
    # PLAN §3.1 names figshare file 3195389 for the tarball.
    assert QM9_FILES[0].figshare_id == 3195389
    assert {f.name for f in QM9_FILES} == {
        "dsgdb9nsd.xyz.tar.bz2", "uncharacterized.txt", "readme.txt",
    }
