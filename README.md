# CE7454 Assignment 1 report (CVPR format)

Built on the official CVPR author kit (`cvpr.sty`, `ieeenat_fullname.bst`, `preamble.tex`
from https://github.com/cvpr-org/author-kit, CVPR 2026 version).

## Compile / Overleaf

Upload this whole folder (or zip it) to Overleaf; set `main.tex` as the main document and
the compiler to pdfLaTeX. Locally:

    pdflatex main && bibtex main && pdflatex main && pdflatex main

`main.tex` uses `\usepackage[pagenumbers]{cvpr}` (camera-ready layout with page numbers).
Switch to `\usepackage[review]{cvpr}` for line numbers.

## Layout (v4 report, 2026-10-16)

- `sec/0_abstract.tex` ... `sec/5_conclusion.tex` - the text, one file per section
- `main.bib` - references
- `fig/overview.tex` - Fig. 1 (training-data pipeline + network), pure TikZ
- `fig/diagnosis.pdf`, `fig/probes.pdf`, `fig/curves_v4.pdf`, `fig/failures.pdf` - Figs. 2-5, from `scripts/fig_v4.py`
- `results/v4_analysis.json`, `results/spectrum_zoom.npz` - numbers behind Figs. 2-5 and Table 1,
  from `scripts/v4_analysis.py` (log: `results/v4_analysis.log`)
- Unused leftovers from the first (baseline-only) report: `fig/npr_examples.png`, `fig/curves.pdf`, `fig/shift.pdf`,
  `fig/robust_table.tex` and the scripts that made them. The pre-v4 sources are not tracked anywhere else.

## Reproducing the v4 figures

From the project root, with the venv and a free GPU (~35 min):

    CUDA_VISIBLE_DEVICES=1 python report/scripts/v4_analysis.py report/results
    cd report && python scripts/fig_v4.py

Test accuracy (95.267%) is the CodaBench score of `results/v4_test/submission.zip`.

## Local compile note

The local TeX Live 2018 lacks `silence`, `cleveref`, a compatible `cuted`, and its old pgf clashes with the `everyshi`
copy inside `cvpr.sty`. Overleaf needs none of this. Locally, put `silence.sty`, `cleveref.sty` (from CTAN) and empty
stubs `cuted.sty` / `everyshi.sty` (`\ProvidesPackage{...}`) in a folder and prepend it to `TEXINPUTS`.
