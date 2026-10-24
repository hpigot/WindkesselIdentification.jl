# Changelog

## 2.0.0 (unreleased)

First release as a registered Julia package. Version 1.0 (git tag `v1.0`) is the code used for
Pigot et al. (2021); this version reproduces its Table 2 (see `test/runtests.jl`).

### Breaking

- The code is a package: `using WindkesselIdentification` replaces `include("oeid.jl")`.
- Units are now mmHg, mL/s and s, as in Stergiopulos et al. (1999). Version 1.0 used flow in
  L/min with time in s, so to convert v1.0 values multiply Rp, Rc and L by 0.06 and divide C by
  0.06. The first state, the volume on C, is now in mL (v1.0 value times 1000/60); the second is
  unchanged. `load_pig` returns flow in mL/s; `Lmin_to_mLs` and `mLs_to_Lmin` convert.
- Hessian condition numbers depend on the units, so they differ from the ones in the paper.
- Scripts moved to `examples/` and data to `data/`. `id_invivo.jl` and `id_exvivo.jl` are now
  `id_pig.jl invivo|exvivo`; `id_stergiopulos1999total.jl` is `id_stergiopulos1999.jl`.
  Multiple initialisation uses a seeded random number generator.

### Fixed

- `solvex0` applied the last input sample before the first, so the initial state was not exactly
  periodic. Eq. (5) of the paper has the same index slip. The published fits move by less than
  0.03%.
- Table 1 of the paper labels C in L/mmHg and L in mmHg min/(L/min), but v1.0 computed them with
  time in seconds: its C = 0.0743 is (L/min) s/mmHg, i.e. 1.24 mL/mmHg.
- The lower-bound vector in `identify` had the wrong length when x0 was identified (harmless:
  the extra bounds were -Inf).

### Changed

- ZOH discretisation in closed form for the diagonal models, replacing `expm.jl` (which overrode
  `Base.exp`) and `c2d`. Simulation and cost are allocation-free, about 30 times faster than v1.0.
- ControlSystems.jl replaced by ControlSystemsBase.jl; GenericLinearAlgebra, Distributions, CSV,
  DataFrames and ToeplitzMatrices are no longer needed.
- `identifyglobal` takes the model as `G_`, like `identify`.
- `spectrum.jl` computes the singular values of the input autocorrelation as the periodogram.

### Added

- `load_stergiopulos1999` and `load_pig` data loaders.
- Tests (including the paper's Table 2 and reference outputs for ports in `test/reference/`),
  benchmarks in `benchmark/`, CI, and figures in `docs/`.
