#!/usr/bin/env python3
"""Build content-addressed VANISH PDF and deterministic artifact ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


FIXED_TIME = (2026, 8, 8, 0, 0, 0)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def included(path: Path, project: Path) -> bool:
    rel = path.relative_to(project)
    parts = set(rel.parts)
    if "__pycache__" in parts or "tmp_render" in parts:
        return False
    if path.suffix in {".pyc", ".aux", ".blg", ".fdb_latexmk", ".fls", ".log", ".out"}:
        return False
    if rel.as_posix() in {"paper/main.pdf", "figures/contact.png"}:
        return False
    if "smoke" in path.name:
        return False
    return True


def zip_bytes(archive: zipfile.ZipFile, arcname: str, data: bytes, executable: bool = False):
    info = zipfile.ZipInfo(arcname, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = ((0o755 if executable else 0o644) & 0xFFFF) << 16
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--paper", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()

    project = args.project.resolve()
    paper = (args.paper or project / "paper/main.pdf").resolve()
    output = (args.output or project.parent / "output").resolve()
    pdf_dir = output / "pdf"
    release_dir = output / "release"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)

    paper_hash = digest(paper)
    paper_name = f"VANISH_IEEE_v{args.version}_{paper_hash[:12]}.pdf"
    paper_out = pdf_dir / paper_name
    shutil.copyfile(paper, paper_out)

    payload: list[tuple[str, bytes, bool]] = []
    for path in sorted(p for p in project.rglob("*") if p.is_file() and included(p, project)):
        rel = path.relative_to(project).as_posix()
        payload.append((rel, path.read_bytes(), path.name == "run_all.sh" or path.name == "build_release.py"))
    payload.append((f"paper/{paper_name}", paper_out.read_bytes(), False))

    metadata = {
        "name": "VANISH",
        "version": args.version,
        "author": "JunHyun Kim",
        "affiliation": "Independent Researcher",
        "release_date": "2026-08-08",
        "paper_filename": paper_name,
        "paper_sha256": paper_hash,
        "functional_runs": 600,
        "mlp_runs": 150,
        "capacity_rows": 200,
        "shape_capacity_runs": 20,
        "unit_contracts": 4,
    }
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    payload.append(("RELEASE.json", metadata_bytes, False))

    manifest_lines = []
    for rel, data, _ in sorted(payload):
        manifest_lines.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
    manifest_bytes = ("\n".join(manifest_lines) + "\n").encode()
    payload.append(("MANIFEST.sha256", manifest_bytes, False))

    with tempfile.TemporaryDirectory(prefix="vanish-release-") as temp:
        provisional = Path(temp) / f"VANISH_ARTIFACT_v{args.version}.zip"
        with zipfile.ZipFile(provisional, "w") as archive:
            for rel, data, executable in sorted(payload):
                zip_bytes(archive, rel, data, executable)
        zip_hash = digest(provisional)
        zip_name = f"VANISH_ARTIFACT_v{args.version}_{zip_hash[:12]}.zip"
        zip_out = release_dir / zip_name
        shutil.copyfile(provisional, zip_out)

    sums_text = (
        f"{paper_hash}  {paper_name}\n"
        f"{zip_hash}  {zip_name}\n"
    )
    sums_hash = hashlib.sha256(sums_text.encode()).hexdigest()
    sums_name = f"VANISH_SHA256SUMS_v{args.version}_{sums_hash[:12]}.txt"
    sums_out = release_dir / sums_name
    sums_out.write_text(sums_text, encoding="utf-8")

    print(json.dumps({
        "paper": str(paper_out),
        "paper_sha256": paper_hash,
        "zip": str(zip_out),
        "zip_sha256": zip_hash,
        "checksums": str(sums_out),
        "checksums_sha256": sums_hash,
        "payload_files": len(payload),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
