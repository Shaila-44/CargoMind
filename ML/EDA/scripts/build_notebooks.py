"""Build the EDA notebooks from their percent-format sources and execute them.

Sources live in eda/notebooks/_source/NN_name.py using `# %%` (code) and `# %% [markdown]`
cell markers, which keeps the analysis diffable. Running this script regenerates every
notebook in eda/notebooks/ with fresh outputs, so the EDA is reproducible end to end.
Run from the CargoMind project root:

    .venv/bin/python eda/scripts/build_notebooks.py            # build + execute all
    .venv/bin/python eda/scripts/build_notebooks.py 02 05      # only notebooks starting 02 / 05
    .venv/bin/python eda/scripts/build_notebooks.py --no-exec  # build without executing
"""
import re
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "_source"
OUT = ROOT / "notebooks"
MARKER = re.compile(r"^# %%(?: \[(markdown)\])?\s*$")


def parse(path):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if not text:
            return
        if kind == "markdown":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in path.read_text().splitlines():
        m = MARKER.match(line)
        if m:
            flush()
            kind, buf = (m.group(1) or "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def main(argv):
    execute = "--no-exec" not in argv
    prefixes = [a for a in argv if not a.startswith("--")]
    for src in sorted(SRC.glob("[0-9][0-9]_*.py")):
        if prefixes and not any(src.name.startswith(p) for p in prefixes):
            continue
        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "cargomind", "display_name": "Python (CargoMind .venv)", "language": "python"}
        nb.cells = parse(src)
        if execute:
            print(f"executing {src.stem} ...", flush=True)
            try:
                ExecutePreprocessor(timeout=900, kernel_name="cargomind").preprocess(nb, {"metadata": {"path": str(OUT)}})
            except Exception:
                nbformat.write(nb, OUT / f"{src.stem}.ipynb")  # keep partial outputs for debugging
                raise
        nbformat.write(nb, OUT / f"{src.stem}.ipynb")
        print(f"wrote eda/notebooks/{src.stem}.ipynb")


if __name__ == "__main__":
    main(sys.argv[1:])
