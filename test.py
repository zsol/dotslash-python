import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


def check_path(dotslash: str, path: Path) -> None:
    contents = path.read_text()
    _header, _, descriptor_text = contents.partition("\n")
    descriptor = json.loads(descriptor_text)
    assert isinstance(descriptor, dict)
    name = descriptor["name"]
    assert isinstance(name, str)
    version = name.removeprefix("cpython-")
    free_threaded = version.endswith("t")
    version = version.removesuffix("t")
    proc = subprocess.run(
        [dotslash, path, "-c", "import sys; print(sys.version)"],
        text=True,
        capture_output=True,
        check=True,
    )
    if not proc.stdout.startswith(version):
        raise AssertionError(f"{version=} not found in {proc.stdout=}")
    is_free_threading_build = "free-threading build" in proc.stdout
    if free_threaded != is_free_threading_build:
        raise AssertionError(f"Requested {free_threaded=}, got {proc.stdout=}")
    print(f"👌{path}")


def main() -> None:
    dotslash = shutil.which("dotslash")
    if dotslash is None:
        print("Dotslash binary cannot be found, failing test.", file=sys.stderr)
        sys.exit(-1)
        return  # TODO(pyrefly): this shouldn't be necessary
    args = sys.argv[1:]
    descriptor_paths: list[Path] = []
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            descriptor_paths.extend(path.glob("*"))
        else:
            descriptor_paths.append(path)
    failure = False
    for path in descriptor_paths:
        try:
            check_path(dotslash, Path(path))
        except Exception:
            failure = True
            traceback.print_exc()
    if failure:
        print("🤔")
        sys.exit(1)
    print("🫡")


if __name__ == "__main__":
    main()
