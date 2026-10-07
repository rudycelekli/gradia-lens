"""A public download must not smuggle private Gradia code or unreviewed bytes."""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.verify_release import VERSION, verify

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    directory = tmp_path_factory.mktemp("lens-build")
    result = subprocess.run(
        [sys.executable, "-m", "build", str(PACKAGE), "--outdir", str(directory)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    verify(directory, PACKAGE)
    return directory


def copy_release(release, tmp_path):
    for path in release.iterdir():
        shutil.copyfile(path, tmp_path / path.name)
    return tmp_path


def rewrite_wheel(directory, transform):
    wheel = next(directory.glob("*.whl"))
    with zipfile.ZipFile(wheel) as source:
        files = [(item.filename, source.read(item.filename)) for item in source.infolist()]
    with zipfile.ZipFile(wheel, "w") as target:
        for name, contents in transform(files):
            target.writestr(name, contents)


def rewrite_source(directory, transform):
    archive_path = next(directory.glob("*.tar.gz"))
    with tarfile.open(archive_path, "r:gz") as source:
        files = [(item, source.extractfile(item).read() if item.isfile() else None) for item in source.getmembers()]
    with tarfile.open(archive_path, "w:gz") as target:
        for item, data in transform(files):
            target.addfile(item, io.BytesIO(data) if data is not None else None)


def test_clean_archives_have_the_declared_portable_boundary(release):
    manifest = verify(release, PACKAGE)
    assert manifest["project"] == "gradia-lens"
    assert manifest["license"] == "Apache-2.0"
    assert manifest["registry_published"] is False
    assert len(manifest["files"]) == 2
    assert "docs/RESEARCH.md" in manifest["source_files"]


def test_private_application_code_is_refused(release, tmp_path):
    directory = copy_release(release, tmp_path)
    rewrite_wheel(directory, lambda files: [*files, ("gradia/db/private.py", b"private application")])
    with pytest.raises(ValueError, match="portable Lens boundary"):
        verify(directory, PACKAGE)


def test_even_a_nonfunctional_engine_change_invalidates_the_release(release, tmp_path):
    directory = copy_release(release, tmp_path)
    rewrite_wheel(
        directory,
        lambda files: [(name, contents + b"\n# changed after review\n" if name == "gradia_lens/engine.py" else contents) for name, contents in files],
    )
    with pytest.raises(ValueError, match="source mismatch: engine.py"):
        verify(directory, PACKAGE)


def test_archive_traversal_is_refused(release, tmp_path):
    directory = copy_release(release, tmp_path)
    rewrite_wheel(directory, lambda files: [*files, ("../private.txt", b"unsafe")])
    with pytest.raises(ValueError, match="Unsafe archive path"):
        verify(directory, PACKAGE)


def test_duplicate_members_cannot_hide_an_alternate_engine(release, tmp_path):
    directory = copy_release(release, tmp_path)
    with pytest.warns(UserWarning, match="Duplicate name"):
        rewrite_wheel(directory, lambda files: [*files, ("gradia_lens/engine.py", b"alternate")])
    with pytest.raises(ValueError, match="Duplicate wheel"):
        verify(directory, PACKAGE)


def test_private_local_receipts_are_refused_in_source_distribution(release, tmp_path):
    directory = copy_release(release, tmp_path)
    item = tarfile.TarInfo(f"gradia_lens-{VERSION}/output/model-research/tenant/private.json")
    contents = b'{"private":true}'
    item.size = len(contents)
    rewrite_source(directory, lambda files: [*files, (item, contents)])
    with pytest.raises(ValueError, match="unreviewed path"):
        verify(directory, PACKAGE)


def test_an_allowlisted_module_still_needs_reviewed_canonical_source(release, tmp_path):
    directory = copy_release(release, tmp_path)
    item = tarfile.TarInfo(f"gradia_lens-{VERSION}/src/gradia_lens/unreviewed.py")
    contents = b"# unreviewed payload\n"
    item.size = len(contents)
    rewrite_source(directory, lambda files: [*files, (item, contents)])
    with pytest.raises(ValueError, match="without canonical source"):
        verify(directory, PACKAGE)


def test_omitting_a_real_engine_dependency_is_refused(release, tmp_path):
    directory = copy_release(release, tmp_path)
    rewrite_source(directory, lambda files: [(item, data) for item, data in files if not item.name.endswith("src/gradia_lens/experiments.py")])
    with pytest.raises(ValueError, match="Missing portable release files"):
        verify(directory, PACKAGE)


@pytest.mark.parametrize("promised", [
    "examples/records/positive-tone.json",
    "examples/records/positive-tone-campaign.json",
    "docs/PAPER_TRACK.md",
    "docs/assets/computation.jpg",
])
def test_promised_research_and_offline_examples_cannot_be_omitted(release, tmp_path, promised):
    directory = copy_release(release, tmp_path)
    rewrite_source(directory, lambda files: [(item, data) for item, data in files if not item.name.endswith("/" + promised)])
    with pytest.raises(ValueError, match="Missing portable release files"):
        verify(directory, PACKAGE)


def test_the_installed_command_cannot_be_redirected_after_review(release, tmp_path):
    directory = copy_release(release, tmp_path)
    rewrite_wheel(
        directory,
        lambda files: [(name, b"[console_scripts]\ngradia-lens = gradia_lens.engine:main\n" if name.endswith("/entry_points.txt") else data) for name, data in files],
    )
    with pytest.raises(ValueError, match="entry point differs"):
        verify(directory, PACKAGE)


def test_generated_build_configuration_cannot_hide_extra_configuration(release, tmp_path):
    directory = copy_release(release, tmp_path)
    contents = b"[egg_info]\ntag_build = \ntag_date = 0\n\n[install]\nextra = unreviewed\n"
    def alter(files):
        for item, data in files:
            if item.name.endswith("/setup.cfg"):
                item.size = len(contents)
                yield item, contents
            else:
                yield item, data
    rewrite_source(directory, alter)
    with pytest.raises(ValueError, match="changes generated setup.cfg"):
        verify(directory, PACKAGE)
