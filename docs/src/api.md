# API

```@docs
WindkesselIdentification
```

The parameter vector is θ = [Rp, C, Rc, L] (the first three for `G3`, the first two for `G2`).

## Models

```@docs
G
G3
G2
simulate
solvex0
```

## Identification

```@docs
identify
identifyglobal
cost
IdConditions
WindkesselIdentification.p
WindkesselIdentification.deal
```

## Data and units

```@docs
load_stergiopulos1999
load_pig
params_stergiopulos1999total
Lmin_to_mLs
mLs_to_Lmin
```
