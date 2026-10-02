"""Execute every tutorial in a fresh kernel without saving bulky outputs."""
import argparse
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--output-dir", type=Path, help="Optional executed copies, outside the source notebooks")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=300, kernel_name=args.kernel,
                                resources={"metadata": {"path": str(root / "notebooks")}})
        client.execute()
        figures = sum("image/png" in output.get("data", {})
                      for cell in notebook.cells for output in cell.get("outputs", []))
        if not figures:
            raise RuntimeError(f"{path.name}: no rendered figures")
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, args.output_dir / path.name)
        print(f"PASS {path.name}", flush=True)


if __name__ == "__main__":
    main()
