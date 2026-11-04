"""Keep bounded, byte-exact pre-BOLT CUDA variants for the optional artifact."""

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sysconfig
from pathlib import Path


def preserve(
    source: Path,
    destination: Path,
    soabi: str,
    native_aot: bool,
    max_variants: int,
) -> None:
    if max_variants < 1:
        raise ValueError("max_variants must be positive")
    digest = hashlib.sha256()
    with source.open("rb") as library:
        for chunk in iter(lambda: library.read(1 << 20), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    manifest_path = destination / "cuda-manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"schema_version": 1, "pythons": {}}
    )
    variants = destination / "cuda"
    variants.mkdir(parents=True, exist_ok=True)
    relative = Path("cuda") / sha256 / "libtorch_cuda.so"
    target = destination / relative
    retained = target.is_file()
    if not retained and len(list(variants.iterdir())) < max_variants:
        target.parent.mkdir()
        staged = target.with_suffix(".tmp")
        try:
            shutil.copyfile(source, staged)
            staged.replace(target)
            retained = True
        finally:
            staged.unlink(missing_ok=True)
            if not retained:
                target.parent.rmdir()
    manifest["pythons"][soabi] = {
        "sha256": sha256,
        "size_bytes": source.stat().st_size,
        "wheel_native_aot": native_aot,
        "path": relative.as_posix() if retained else None,
        "status": "retained" if retained else "variant_limit_exceeded",
    }
    staged_manifest = manifest_path.with_suffix(".tmp")
    staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    staged_manifest.replace(manifest_path)
    if not retained:
        print(
            f"::warning::pre-BOLT CUDA variant limit ({max_variants}) exceeded "
            f"for {soabi}; see cuda-manifest.json"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--max-variants",
        type=int,
        default=int(os.environ.get("PREBOLT_CUDA_MAX_VARIANTS", "2")),
    )
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f"missing pre-BOLT CUDA library: {args.source}")
    # build.sh installed this iteration's wheel and patched it after stage 2.
    # This records the wheel's AOT state, not a probe of the archived library.
    torch = importlib.import_module("torch")
    preserve(
        args.source,
        args.destination,
        sysconfig.get_config_var("SOABI"),
        bool(torch._native._native_aot_embedded()),
        args.max_variants,
    )


if __name__ == "__main__":
    main()
