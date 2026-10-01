# Document builds

- After changing paper or thesis sources, always compile the affected document and update its PDF before finishing.
- Run the bibliography and repeat LaTeX passes as needed to resolve references. Fix build errors and report any remaining warnings.
- The local TeX installation is at `/home/umer/.TinyTeX/bin/x86_64-linux`; prepend this directory to `PATH` when the compiler is unavailable.
- For `EG2027/3dhsi`, build `paper.tex` with pdfLaTeX, run BibTeX, then repeat pdfLaTeX until references stabilize.
- Rebuild the 3dhsi supplement with `python3 EG2027/3dhsi/build_supplementary.py` from the repository root. Its LaTeX source is `EG2027/3dhsi/supplementary.tex`; retain all six original prompt templates and the main paper's qualitative figure styling.
