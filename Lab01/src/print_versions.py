from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_FILE = REPOSITORY_ROOT / "requirements.txt"


def main() -> None:
    versions = [f"Python=={platform.python_version()}"]
    for requirement in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines():
        requirement = requirement.strip()
        if not requirement or requirement.startswith("#"):
            continue
        package = requirement.split("==", maxsplit=1)[0]
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Required package is not installed: {package}") from error
        versions.append(f"{package}=={version}")

    output = "\n".join(versions) + "\n"
    results_dir = REPOSITORY_ROOT / "Lab01" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "versions.txt").write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
