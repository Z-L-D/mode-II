# Projection, Pooling and Tangent Agreement: Equivalence Relations for Layerwise Spectral Trajectories of Generative Transformers (Mode-II)

Zac Doyle, mode:LIMINAL. Repository: https://github.com/Z-L-D/mode-II

## Layout
- `main.tex`: the paper (arXiv-compatible: pdflatex + bibtex, natbib/plainnat).
- `supplement.tex`: the Supplementary material, `\input` at the end of `main.tex` so that cross-references resolve; it is numbered S1, S2, ... and the release also ships it as a separate PDF.
- `references.bib`: bibliography. Every entry was checked against a primary page; see `LIT_REVIEW.md` and `CITATION_KEYS.md` in the parent folder.
- `figures/`: one PDF (used by LaTeX) and one PNG per figure, plus `CAPTIONS.md` (caption, scope and provenance of every figure).
- `scripts/`: one Python script per figure and `make_all_figures.py`. Two scripts check the mathematics numerically and need no telemetry: `pooling_bound_check.py` (the pooling theorem) and `lemma_rotation_invariance.py` (the rotation-invariance lemma, its corollary on network symmetries, and the counterexample to sufficiency; writes `data/II-lemma_rotation_check.csv`).
- `data/`: the exact plotted values for each figure, with a `source` column naming where each number comes from.
- `../common/figstyle.py`: shared plotting style (keep it next to this folder).

## Build
    pdflatex main && bibtex main && pdflatex main && pdflatex main

## Rebuild the figures
    python3 scripts/make_all_figures.py            # all figures
    python3 scripts/make_all_figures.py --only <id>

Requires Python 3 with numpy, pandas, scipy and matplotlib. Figures that recompute from raw telemetry need the analysis cubes: unzip the derived bundle at the repository root (see ../../DATA.md); `SCR` and `REVIEW` override the defaults `<root>/cache` and `<root>/review`. Every other figure reads its inputs from those outputs and `data/`.
