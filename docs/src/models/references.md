# [References](@id references)

References are maintained here as author–year entries with persistent links. Theory pages link
to the source that supports each literature claim. Equations derived from the implementation
are labeled as such. Numerical agreement within this package is distinguished from independent
benchmark evidence.

- Jech, J. M., et al. (2015). *Comparisons among ten models of acoustic backscattering used in
  aquatic ecosystem research*. Journal of the Acoustical Society of America, 138, 3742–3764.
  [DOI: 10.1121/1.4937607](https://doi.org/10.1121/1.4937607).
  Canonical model comparisons, including sphere, shell, spheroid, and finite cylinder.
- Furusawa, M. (1988). *Prolate spheroidal models for predicting general trends of fish target
  strength*. Journal of the Acoustical Society of Japan (E), 9, 13–24.
  [Publisher article](https://www.jstage.jst.go.jp/article/ast1980/9/1/9_1_13/_article).
  Spheroidal model assumptions and angular coupling.
- Goodman, R. R., and Stern, R. (1962). *Reflection and transmission of sound by elastic
  spherical shells*. Journal of the Acoustical Society of America.
  [DOI: 10.1121/1.1928120](https://doi.org/10.1121/1.1928120).
  Elastic-shell interface formulation. Distinguish the original identical-fluid assumption
  from the package's generalized interior-fluid treatment.
- Hickling, R. (1962). *Analysis of echoes from a solid elastic sphere in water*.
  Journal of the Acoustical Society of America.
  [DOI: 10.1121/1.1909055](https://doi.org/10.1121/1.1909055).
  Solid-elastic sphere scattering.
- Fairweather, G., Karageorghis, A., and Martin, P. A. (2003).
  *The method of fundamental solutions for scattering and radiation problems*.
  Engineering Analysis with Boundary Elements.
  [DOI: 10.1016/S0955-7997(03)00017-1](https://doi.org/10.1016/S0955-7997(03)00017-1).
  [Author-hosted article](https://people.mines.edu/pamartin/wp-content/uploads/sites/254/2024/06/R080_EABE.pdf).
  General MFS background, not a validation of package-specific source placement.

The source also attributes viscous-elastic scattering to Feuillade and Nero (1998), bent-body
coherence corrections to Stanton, and the thin-shell formulation to Hayek and Boisvert (2003).
Complete equation-level citation verification for those branches is pending. The repository's
`docs/NUMERICAL_METHODS_LITERATURE_MAP.md` is a research inventory, not a list of models
already implemented in the package.

For tooling:
[Documenter executable blocks](https://documenter.juliadocs.org/stable/man/syntax/),
[PrecompileTools preferences](https://julialang.github.io/PrecompileTools.jl/stable/),
and [SciML style](https://docs.sciml.ai/SciMLStyle/stable/).
