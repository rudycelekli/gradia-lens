"""Verify a portable Lens source/wheel boundary before staging local downloads.

This checks archive structure and source identity. It does not publish a release,
run model evaluations, or attest that the package exists on a registry.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import stat
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0a1"
PROJECT = "gradia-lens"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def wheel_allowed(path: PurePosixPath) -> bool:
    parts = path.parts
    if len(parts) == 2 and parts[0] == "gradia_lens":
        return path.suffix == ".py" or parts[1] == "py.typed"
    metadata = f"gradia_lens-{VERSION}.dist-info"
    if parts and parts[0] == metadata:
        return str(path.relative_to(metadata)) in {
            "METADATA", "WHEEL", "entry_points.txt", "top_level.txt", "RECORD", "licenses/LICENSE", "licenses/NOTICE",
        }
    return False


def source_allowed(path: PurePosixPath) -> bool:
    if len(path.parts) == 1:
        return path.name in {
            "pyproject.toml", "README.md", "LICENSE", "NOTICE", "MANIFEST.in", "PKG-INFO",
            "CONTRIBUTING.md", "CHANGELOG.md", "SECURITY.md", "setup.cfg",
        }
    if len(path.parts) == 3 and path.parts[:2] == ("src", "gradia_lens"):
        return path.suffix == ".py" or path.name == "py.typed"
    if len(path.parts) == 3 and path.parts[:2] == ("src", "gradia_lens.egg-info"):
        return path.name in {"PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt", "requires.txt", "top_level.txt"}
    if str(path) == "docs/assets/computation.jpg":
        return True
    if str(path) == "examples/records/positive-tone.json":
        return True
    if len(path.parts) == 3 and path.parts[:2] == ("docs", "adr") and path.suffix == ".md":
        return True
    return len(path.parts) == 2 and (
        (path.parts[0] in {"examples", "tests", "scripts"} and path.suffix == ".py")
        or (path.parts[0] == "docs" and path.suffix == ".md")
    )


def verify(dist: Path, package: Path) -> dict:
    wheels = list(dist.glob(f"gradia_lens-{VERSION}-*.whl"))
    sources = list(dist.glob(f"gradia_lens-{VERSION}.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one Lens alpha wheel and source distribution")
    wheel, source = wheels[0], sources[0]
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate wheel archive entries")
        if any(stat.S_ISLNK(item.external_attr >> 16) for item in archive.infolist()):
            raise ValueError("Wheel contains a symbolic link")
        if any(not wheel_allowed(safe_path(name)) for name in names):
            raise ValueError("Wheel contains files outside the portable Lens boundary")
        metadata = BytesParser().parsebytes(archive.read(f"gradia_lens-{VERSION}.dist-info/METADATA"))
        if metadata["Name"] != PROJECT or metadata["Version"] != VERSION or metadata["License-Expression"] != "Apache-2.0":
            raise ValueError("Unexpected package identity or license metadata")
        requirements = metadata.get_all("Requires-Dist", [])
        if any("extra ==" not in item for item in requirements):
            raise ValueError("Base Lens install must have no runtime dependencies")
        model_requirements = {
            item.split(";", 1)[0].strip()
            for item in requirements
            if item.split(";", 1)[1].replace(" ", "").replace("'", '"') == 'extra=="model"'
        }
        if model_requirements != {
            "torch==2.8.0", "transformers==4.57.3", "numpy==1.26.4",
            "huggingface-hub==0.36.0", "tokenizers==0.22.1", "safetensors==0.7.0",
        } or metadata["Requires-Python"] != ">=3.9":
            raise ValueError("Wheel compatibility metadata differs from the verified edition")
        module_names = {str(Path(name).name) for name in names if name.startswith("gradia_lens/")}
        expected_modules = {p.name for p in (package / "src/gradia_lens").iterdir() if p.is_file() and (p.suffix == ".py" or p.name == "py.typed")}
        if module_names != expected_modules:
            raise ValueError("Wheel module set differs from the canonical source")
        for name in module_names:
            canonical = package / "src/gradia_lens" / name
            if canonical.is_symlink() or archive.read(f"gradia_lens/{name}") != canonical.read_bytes():
                raise ValueError(f"Wheel source mismatch: {name}")
        if archive.read(f"gradia_lens-{VERSION}.dist-info/licenses/LICENSE") != (package / "LICENSE").read_bytes():
            raise ValueError("Wheel license differs from the declared package license")
        if archive.read(f"gradia_lens-{VERSION}.dist-info/licenses/NOTICE") != (package / "NOTICE").read_bytes():
            raise ValueError("Wheel attribution differs from the declared package notice")
        entry_points = configparser.ConfigParser()
        entry_points.read_string(archive.read(f"gradia_lens-{VERSION}.dist-info/entry_points.txt").decode())
        if entry_points.sections() != ["console_scripts"] or dict(entry_points["console_scripts"]) != {"gradia-lens": "gradia_lens.cli:main"}:
            raise ValueError("Wheel command entry point differs from the reviewed CLI")
    source_files = []
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        if len({member.name for member in members}) != len(members):
            raise ValueError("Duplicate source archive entries")
        prefix = f"gradia_lens-{VERSION}"
        for member in members:
            path = safe_path(member.name)
            if path.parts[0] != prefix or member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise ValueError("Unsafe source archive member")
            if member.isdir():
                continue
            relative = path.relative_to(prefix)
            if not source_allowed(relative):
                raise ValueError(f"Source distribution includes an unreviewed path: {relative}")
            source_files.append(str(relative))
            if str(relative) == "setup.cfg":
                stream = archive.extractfile(member)
                if stream is None or stream.read() != b"[egg_info]\ntag_build = \ntag_date = 0\n\n":
                    raise ValueError("Source distribution changes generated setup.cfg")
                continue
            canonical = package / str(relative)
            generated = str(relative) == "PKG-INFO" or "gradia_lens.egg-info" in relative.parts
            if not generated:
                if not canonical.is_file() or canonical.is_symlink():
                    raise ValueError(f"Source distribution includes a file without canonical source: {relative}")
                stream = archive.extractfile(member)
                if stream is None or stream.read() != canonical.read_bytes():
                    raise ValueError(f"Source distribution differs from canonical file: {relative}")
        required = {"pyproject.toml", "README.md", "LICENSE", "NOTICE", "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md", "docs/RESEARCH.md"}
        required.update(f"src/gradia_lens/{name}" for name in expected_modules)
        if not required.issubset(source_files):
            raise ValueError(f"Missing portable release files: {sorted(required - set(source_files))}")
    modules = package / "src/gradia_lens"
    engine_hash = hashlib.sha256()
    for name in ("engine.py", "experiments.py"):
        engine_hash.update((modules / name).read_bytes())
    return {
        "schema": "gradia.lens.release.v1", "project": PROJECT, "version": VERSION,
        "license": "Apache-2.0", "status": "prepared_local_alpha", "registry_published": False,
        "verification": "Archive boundary and exact canonical source bytes verified; execution checks recorded separately.",
        "engine_source_sha256": engine_hash.hexdigest(),
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p.read_bytes())} for p in (wheel, source)],
        "source_files": sorted(source_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    parser.add_argument("--package", type=Path, default=ROOT)
    args = parser.parse_args()
    manifest = verify(args.dist, args.package)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
