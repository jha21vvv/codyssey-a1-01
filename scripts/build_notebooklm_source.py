"""Consolidate pure production source files into NOTEBOOKLM_FULL_SOURCE.txt."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "NOTEBOOKLM_FULL_SOURCE.txt"

SOURCE_FILES = [
    "src/__init__.py",
    "src/pipeline.py",
    "src/visualize.py",
    "scripts/prepare_dataset.py",
    "scripts/generate_diagrams_matplotlib.py",
    "scripts/verify_requirements.py",
]


def main() -> None:
    chunks: list[str] = []
    for rel_path in SOURCE_FILES:
        file_path = ROOT / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"소스 파일을 찾을 수 없습니다: {file_path}")
        content = file_path.read_text(encoding="utf-8")
        header = f"{'=' * 72}\nFILE: {rel_path}\n{'=' * 72}\n"
        chunks.append(header + content.rstrip() + "\n")

    OUT_FILE.write_text("\n".join(chunks), encoding="utf-8")
    print(f"Generated {OUT_FILE.name} from {len(SOURCE_FILES)} source files ({OUT_FILE.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
