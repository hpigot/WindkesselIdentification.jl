"""
Identification of 2-, 3- and parallel 4-element Windkessel models from measured aortic pressure and flow.

Units throughout: pressure in mmHg, flow in mL/s, time in s, so that
Rp, Rc [mmHg s/mL], C [mL/mmHg], L [mmHg s^2/mL].
"""
module WindkesselIdentification

using LinearAlgebra, DelimitedFiles, ControlSystemsBase, Optim, LineSearches, ForwardDiff

export IdConditions, identify, identifyglobal, cost, simulate, G, G3, G2, solvex0,
    params_stergiopulos1999total, mLs_to_Lmin, Lmin_to_mLs, load_stergiopulos1999, load_pig

"""
    mLs_to_Lmin(q)

Convert flow `q` from mL/s to L/min. Use broadcasting (`mLs_to_Lmin.(q)`) for vectors.

# Examples
```jldoctest
julia> mLs_to_Lmin(100)
6.0
```
"""
mLs_to_Lmin(q) = q * 60 / 1000

"""
    Lmin_to_mLs(q)

Convert flow `q` from L/min to mL/s. Use broadcasting (`Lmin_to_mLs.(q)`) for vectors.

# Examples
```jldoctest
julia> Lmin_to_mLs(6)
100.0
```
"""
Lmin_to_mLs(q) = q * 1000 / 60

# Continuous-time models in diagonal form: x' = a.*x + b*u, y = c'x + d*u, as tuples (a, b, c, d).
# The state x1 is the volume on C [mL]; for the 4-element model x2 = L*i_L, with i_L the flow through L.
"Parallel 4-element Windkessel (Burattini, Westerhof, Stergiopulos), θ = [Rp, C, Rc, L]."
wk4(θ) = ((-1 / (θ[2] * θ[1]), -θ[3] / θ[4]), (one(θ[1]), θ[3]), (1 / θ[2], -θ[3] / θ[4]), θ[3])
"3-element Windkessel, θ = [Rp, C, Rc]. The 4-element model with L = Inf."
wk3(θ) = ((-1 / (θ[2] * θ[1]),), (one(θ[1]),), (1 / θ[2],), θ[3])
"2-element Windkessel, θ = [Rp, C]. The 4-element model with Rc = 0 (L irrelevant)."
wk2(θ) = ((-1 / (θ[2] * θ[1]),), (one(θ[1]),), (1 / θ[2],), zero(θ[1]))

"Exact ZOH discretisation of a diagonal model: returns (ad, bd, c, d)."
function zoh((a, b, c, d), h)
    ad = map(ai -> exp(ai * h), a)
    bd = map((adi, ai, bi) -> (adi - 1) / ai * bi, ad, a, b)
    return ad, bd, c, d
end

function statespace(m, h, discretize)
    a, b, c, d = discretize ? zoh(m, h) : m
    sys = (A = diagm(collect(a)), B = reshape(collect(b), :, 1), C = reshape(collect(c), 1, :), D = fill(d, 1, 1))
    return discretize ? ss(sys.A, sys.B, sys.C, sys.D, h) : ss(sys.A, sys.B, sys.C, sys.D)
end

"""
    G(θ, h; discretize=true) -> StateSpace

Parallel 4-element Windkessel model (Burattini, Westerhof, Stergiopulos) from aortic
flow [mL/s] to aortic pressure [mmHg], as a ControlSystemsBase `StateSpace` with 2 states.

# Arguments
- `θ`: parameters `[Rp, C, Rc, L]`: peripheral resistance Rp [mmHg s/mL], compliance
  C [mL/mmHg], characteristic resistance Rc [mmHg s/mL], inertance L [mmHg s^2/mL].
- `h`: sampling period [s] (ignored when `discretize=false`).

# Keyword arguments
- `discretize=true`: if `true`, return the exact zero-order-hold discretisation with
  period `h`; if `false`, return the continuous-time model.

The state is diagonal: x1 = volume on C [mL], x2 = L times the flow through L [mmHg s].

# Examples
```jldoctest
julia> sys = G([0.79, 1.22, 0.056, 0.0051], 0.005);

julia> sys.nx, sys.nu, sys.ny
(2, 1, 1)
```
"""
G(θ, h; discretize=true) = statespace(wk4(θ), h, discretize)

"""
    G3(θ, h; discretize=true) -> StateSpace

3-element Windkessel model from aortic flow [mL/s] to aortic pressure [mmHg], with 1 state.
Equals the 4-element model [`G`](@ref) with L = Inf.

# Arguments
- `θ`: parameters `[Rp, C, Rc]` with Rp, Rc [mmHg s/mL] and C [mL/mmHg]. Only the first
  3 entries are used.
- `h`: sampling period [s] (ignored when `discretize=false`).

# Keyword arguments
- `discretize=true`: ZOH-discretise with period `h` if `true`, else continuous time.

The state x1 is the volume on C [mL].
"""
G3(θ, h; discretize=true) = statespace(wk3(θ), h, discretize)

"""
    G2(θ, h; discretize=true) -> StateSpace

2-element Windkessel model from aortic flow [mL/s] to aortic pressure [mmHg], with 1 state.
Equals the 4-element model [`G`](@ref) with Rc = 0 (L then has no effect).

# Arguments
- `θ`: parameters `[Rp, C]` with Rp [mmHg s/mL] and C [mL/mmHg]. Only the first 2
  entries are used.
- `h`: sampling period [s] (ignored when `discretize=false`).

# Keyword arguments
- `discretize=true`: ZOH-discretise with period `h` if `true`, else continuous time.

The state x1 is the volume on C [mL].
"""
G2(θ, h; discretize=true) = statespace(wk2(θ), h, discretize)

# Map a model constructor (G, G3, G2) to its diagonal form, so `IdConditions.G_` keeps working.
ctmodel(::typeof(G), θ) = wk4(θ)
ctmodel(::typeof(G3), θ) = wk3(θ)
ctmodel(::typeof(G2), θ) = wk2(θ)

"Initial state as a tuple of the model's length and of the type the recursion will produce (type-stable loops)."
function initstate((ad, bd, c, d)::Tuple{NTuple{N,Any},Any,Any,Any}, x0, u) where {N}
    T = promote_type(eltype(ad), eltype(bd), eltype(c), typeof(d), eltype(x0), eltype(u))
    return ntuple(i -> T(x0[i]), Val(N))
end

"Simulate discrete diagonal model `md` from initial state `x0`; writes the output into `y` (and states into `X`, if given)."
function sim!(y, md, u, x0; X=nothing)
    ad, bd, c, d = md
    x = initstate(md, x0, u)
    for k in eachindex(u, y)
        X === nothing || (X[k, :] .= x)
        y[k] = sum(map(*, c, x)) + d * u[k]
        x = map((adi, xi, bi) -> adi * xi + bi * u[k], ad, x, bd)
    end
    return y
end

"Sum of squared output errors of discrete diagonal model `md`, without allocating."
function sse(md, u, y, x0)
    ad, bd, c, d = md
    x = initstate(md, x0, u)
    s = zero(promote_type(eltype(x), eltype(y)))
    for k in eachindex(u, y)
        e = y[k] - (sum(map(*, c, x)) + d * u[k])
        s += e * e
        x = map((adi, xi, bi) -> adi * xi + bi * u[k], ad, x, bd)
    end
    return s
end

"""
    solvex0(sys, u) -> x0

Initial state for which the response to the input `u` is periodic: x(n) = x(0), with
n = length(u). With s the state reached from x(0) = 0 after n steps, x(0) = (I - A^n)^-1 s.

# Arguments
- `sys`: a discrete-time `StateSpace`, e.g. `G(θ, h)`, `G3(θ, h)` or `G2(θ, h)`.
- `u`: input vector, aortic flow [mL/s], typically one or more whole beats.

# Returns
- `x0`: vector of length `sys.nx`, in the model's state units (x1 volume on C [mL];
  for `G` also x2 = L times the flow through L [mmHg s]).

[`cost`](@ref) uses this at every evaluation when `x0solve=true`.
"""
function solvex0(md::Tuple, u::AbstractVector)
    ad, bd, _, _ = md
    s = initstate(md, zeros(length(ad)), u)
    for k in eachindex(u)
        s = map((adi, si, bi) -> adi * si + bi * u[k], ad, s, bd)
    end
    return collect(map((adi, si) -> si / (1 - adi^length(u)), ad, s))
end

function solvex0(sys::StateSpace{<:Discrete}, u::AbstractVector)
    x = lsim(sys, reshape(u, 1, :); x0=zeros(sys.nx)).x
    s = sys.A * x[:, end] + vec(sys.B) * u[end]
    return (I - sys.A^length(u)) \ s
end

"""
    IdConditions(θ0, x00, θid, x0id, x0solve, G_=G)

Identification setup: which parameters are free, their fixed or initial values, and the
model. Built internally by [`identify`](@ref) and [`identifyglobal`](@ref); construct it
yourself to call [`cost`](@ref) or `WindkesselIdentification.deal`.

# Fields
- `θ0::Vector{Float64}`: model parameters `[Rp, C, Rc, L]` (first 3 for `G3`, first 2 for
  `G2`); Rp, Rc [mmHg s/mL], C [mL/mmHg], L [mmHg s^2/mL]. Initial guess for free entries,
  fixed value for the others.
- `x00::Vector{Float64}`: initial state x(0) (x1 volume on C [mL]; for `G` also
  x2 = L times the flow through L [mmHg s]). Initial guess if `x0id`, else fixed.
- `θid::Vector{Int}`: 0/1 mask, same length as `θ0`; 1 marks a free (estimated) entry,
  0 keeps it fixed at `θ0`. E.g. `[1,1,1,0]` fixes L.
- `x0id::Bool`: estimate the initial state as extra parameters.
- `x0solve::Bool`: recompute x(0) from periodicity with [`solvex0`](@ref) at every cost
  evaluation (use with `x0id=false`).
- `G_::Function`: model constructor, [`G`](@ref), [`G3`](@ref) or [`G2`](@ref).

The 5-argument constructor sets `G_ = G`. Throws an error if `length(θ0) != length(θid)`.
"""
struct IdConditions
    θ0::Array{Float64,1}    # [Rp,C,Rc,L]
    x00::Array{Float64,1}
    θid::Array{Int,1}       # e.g. [1,1,1,0]
    x0id::Bool              # identify x0
    x0solve::Bool # solve for x0 using its periodicity at every optimization step
    G_::Function # model constructor: G, G3 or G2
    IdConditions(θ0,x00,θid,x0id,x0solve,G_) = (length(θ0) == length(θid)) ? new(θ0,x00,θid,x0id,x0solve,G_) : error("Length of `θ0` != `θid`.")
end

IdConditions(θ0, x00, θid, x0id, x0solve) = IdConditions(θ0,x00,θid,x0id,x0solve,G) # default to 4-element windkessel model, G

"""
    cost(p, id::IdConditions, u, y, h) -> Real

Sum of squared output errors sum((y - yhat).^2) [mmHg^2] of the model `id.G_`,
ZOH-discretised with period `h`, driven by `u`. This is the objective minimised by
[`identify`](@ref) and [`identifyglobal`](@ref).

# Arguments
- `p`: optimisation vector: the free entries of θ (where `id.θid` is 1, in order), followed
  by x(0) if `id.x0id`. Build it with `WindkesselIdentification.p(θ, x0, id)`.
- `id`: [`IdConditions`](@ref); fixed θ entries and (unless `x0id`) x(0) come from it.
  If `id.x0solve`, x(0) is instead computed with [`solvex0`](@ref).
- `u`: input, aortic flow [mL/s].
- `y`: measured output, aortic pressure [mmHg], same length as `u`.
- `h`: sampling period [s].

Divide by `length(u)` for the mean squared error.
"""
function cost(p,id::IdConditions,u,y,h)
    θ,x0 = deal(p,id)
    md = zoh(ctmodel(id.G_, θ), h)
    if id.x0solve x0 = solvex0(md,u) end
    return sse(md, u, y, x0)
end

"""
    simulate(θ, x0, u, h; xout=false, G_=G) -> y  or  (y, X)

Simulate the ZOH-discretised model `G_(θ, h)` with input `u` from initial state `x0`.

# Arguments
- `θ`: parameters `[Rp, C, Rc, L]` (first 3 for `G3`, first 2 for `G2`); Rp, Rc
  [mmHg s/mL], C [mL/mmHg], L [mmHg s^2/mL].
- `x0`: initial state (x1 volume on C [mL]; for `G` also x2 = L times the flow through L
  [mmHg s]).
- `u`: input vector, aortic flow [mL/s].
- `h`: sampling period [s].

# Keyword arguments
- `xout=false`: also return the state trajectory.
- `G_=G`: model, [`G`](@ref), [`G3`](@ref) or [`G2`](@ref).

# Returns
- `y`: aortic pressure [mmHg], same length as `u`.
- `X` (only if `xout=true`): `length(u)` x `length(x0)` matrix, row k is the state at
  sample k (before applying `u[k]`).
"""
function simulate(θ,x0,u,h; xout=false, G_=G)
    md = zoh(ctmodel(G_, θ), h)
    y = similar(u, promote_type(eltype(u), eltype(θ)))
    X = xout ? similar(y, length(u), length(x0)) : nothing
    sim!(y, md, u, x0; X=X)
    return xout ? (y, X) : y
end

"""
    identify(θ0, x00, u, y, h; θid=[1,1,1,1], x0id=true, x0solve=false, G_=G, θpos=true, v=false) -> (phat, res)

Estimate a chosen subset of the model parameters θ and the initial state x(0) by
minimising [`cost`](@ref), the sum of squared output errors, with second-order Optim
methods (exact ForwardDiff gradient and Hessian).

# Arguments
- `θ0`: initial guess `[Rp, C, Rc, L]` (first 3 for `G3`, first 2 for `G2`); Rp, Rc
  [mmHg s/mL], C [mL/mmHg], L [mmHg s^2/mL]. Entries not marked free in `θid` stay fixed
  at these values.
- `x00`: initial guess for x(0) (x1 volume on C [mL]; for `G` also x2 = L times the flow
  through L [mmHg s]); used as is when `x0id=false` and `x0solve=false`.
- `u`: input, aortic flow [mL/s].
- `y`: measured output, aortic pressure [mmHg], same length as `u`.
- `h`: sampling period [s].

# Keyword arguments
- `θid=[1,1,1,1]`: 0/1 mask over θ; 1 = estimated, 0 = fixed at `θ0`. Must have the same
  length as `θ0`, so pass e.g. `θid=[1,1,1]` with `G_=G3`.
- `x0id=true`: also estimate x(0), as extra parameters after the free θ entries.
- `x0solve=false`: recompute x(0) at every cost evaluation as the periodic initial state
  from [`solvex0`](@ref), instead of using `x00` or estimating it. Use with `x0id=false`.
- `G_=G`: model, [`G`](@ref) (4-element), [`G3`](@ref) or [`G2`](@ref).
- `θpos=true`: if `true`, use Optim `IPNewton` with bounds θ >= 0 on the free θ entries
  (x(0) unbounded); if `false`, use unconstrained `Newton` with a backtracking line search.
- `v=false`: show the Optim trace.

# Returns
- `phat`: estimated optimisation vector: the free θ entries (in order), then x(0) if
  `x0id`. Use `WindkesselIdentification.deal(phat, IdConditions(θ0, x00, θid, x0id, x0solve, G_))`
  to get the full `(θ, x0)` back.
- `res`: the Optim result. `res.minimum` is the sum of squared errors [mmHg^2]; divide by
  `length(u)` for the MSE.

See also [`identifyglobal`](@ref).
"""
function identify(θ0,x00,u,y,h;θid=[1,1,1,1],x0id=true,x0solve=false,G_=G,θpos=true,v=false)
    id = IdConditions(θ0,x00,θid,x0id,x0solve,G_)
    p0 = p(θ0,x00,id)
    f(p) = cost(p,id,u,y,h)
    g!(grad,p) = ForwardDiff.gradient!(grad,f,p)
    function h!(hess,p)
        ForwardDiff.hessian!(hess,f,p)
        hess .= (hess .+ hess') ./ 2 # symmetrise
    end

    df = TwiceDifferentiable(f,g!,h!,p0)
    opts = Optim.Options(store_trace = true, extended_trace = true, show_trace = v)
    local res
    if θpos
        up = Float64[]
        lp = [fill(0.0, sum(id.θid)); fill(-Inf, length(p0) - sum(id.θid))] # θ >= 0, x0 free
        dfc = TwiceDifferentiableConstraints(lp, up)
        res = optimize(df,dfc,p0,IPNewton(),opts)
    else
        res = optimize(df,p0,Newton(alphaguess = LineSearches.InitialPrevious(), linesearch = LineSearches.BackTracking()),opts)
    end

    phat = Optim.minimizer(res)
    return phat,res
end

"""
    identifyglobal(θ0, x00, u, y, h; θid=[1,1,1,1], x0id=true, x0solve=false, G_=G, v=false) -> (phat, res)

Same problem as [`identify`](@ref), solved with the derivative-free Optim `NelderMead`
(up to 1e5 iterations). Less sensitive to the initial guess, but slower.

The search is unconstrained: unlike `identify` with `θpos=true` there is no positivity
constraint on θ, so check the signs of the estimate.

Arguments, keyword arguments and returns are as for [`identify`](@ref) (there is no `θpos`).
Get `(θ, x0)` with `WindkesselIdentification.deal(phat, IdConditions(...))`; `res.minimum`
is the sum of squared errors [mmHg^2].
"""
function identifyglobal(θ0,x00,u,y,h;θid=[1,1,1,1],x0id=true,x0solve=false,G_=G,v=false)
    id = IdConditions(θ0,x00,θid,x0id,x0solve,G_)
    p0 = p(θ0,x00,id)
    f(p) = cost(p,id,u,y,h)
    opts = Optim.Options(iterations=Int(1e5),store_trace = true, extended_trace = true, show_trace = v)
    res = optimize(f, p0, NelderMead(),opts) # FIXME: constrain to θ > 0

    phat = Optim.minimizer(res)
    return phat,res
end

"""
    WindkesselIdentification.p(θ, x0, id::IdConditions) -> Vector{Float64}

Build the optimisation vector from the full parametrisation: the entries of `θ` where
`id.θid` is 1 (in order), followed by `x0` if `id.x0id`. Inverse of
`WindkesselIdentification.deal`. Not exported.
"""
function p(θ,x0,id::IdConditions)
    p = θ[findall(Bool.(id.θid))]
    if id.x0id append!(p,x0) end
    return Float64.(p)
end

"""
    WindkesselIdentification.deal(p, id::IdConditions) -> (θ, x0)

Map an optimisation vector `p` (e.g. `phat` from [`identify`](@ref)) back to the full
parametrisation: free θ entries are taken from `p`, fixed ones from `id.θ0`; `x0` is taken
from `p` if `id.x0id`, else it is `id.x00`. Inverse of `WindkesselIdentification.p`.
Not exported.

Note: with `x0solve=true` the returned `x0` is `id.x00`, not the periodic state used in
the cost; recompute it with [`solvex0`](@ref).
"""
function deal(p,id::IdConditions)
    pidx = cumsum(id.θid)
    θ = [Bool(id_) ? p[k] : θ0_ for (k,θ0_,id_) in zip(pidx,id.θ0,id.θid)] # p[k] only for free entries: k is 0 before the first
    x0 = id.x0id ? p[sum(id.θid).+(1:length(id.x00))] : id.x00
    return θ,x0
end

"""
    params_stergiopulos1999total() -> (θ, x0)

Reference parameters of the 4-element model from Stergiopulos et al. (1999), Table 1,
type A beat, for use with [`load_stergiopulos1999`](@ref).

# Returns
- `θ = [Rp, C, Rc, L] = [0.79, 1.22, 0.056, 0.0051]`: Rp, Rc [mmHg s/mL], C [mL/mmHg],
  L [mmHg s^2/mL].
- `x0`: steady-state (periodic) initial state for [`G`](@ref): x1 [mL], x2 [mmHg s].
"""
function params_stergiopulos1999total()
    Rp = .79                # peripheral (systemic) resistance [mmHg*ml^-1*s]
    C = 1.22                # total arterial compliance [ml/mmHg]
    Rc = .056               # characteristic aortic resistance [mmHg*ml^-1*s]
    L = .0051               # total arterial inertance [mmHg*s^2*ml^-1]
    θ = [Rp,C,Rc,L]
    x0 = [8.066*1000/60, -8.296] # steady-state sim results, converted from the v1.0 flow unit (L/min)
    return θ, x0
end

const DATADIR = joinpath(dirname(@__DIR__), "data")

"""
    load_stergiopulos1999() -> (p, q, h)

One type A beat from Stergiopulos et al. (1999), Figure 4A, digitised at 200 Hz:
aortic pressure `p` [mmHg] and flow `q` [mL/s], sampling period `h` [s].
"""
function load_stergiopulos1999()
    d = joinpath(DATADIR, "stergiopulos1999_data")
    p = readdlm(joinpath(d, "apressure.csv"), ',', Float64)[:, 2]
    q = readdlm(joinpath(d, "aflow.csv"), ',', Float64)[:, 2]
    return p, q, 0.005
end

"""
    load_pig(source::Symbol; segments=1:4) -> (p, q, h)

Aortic pressure `p` [mmHg] and flow `q` [mL/s] from 65 kg pigs, `source` = `:invivo`
(2020-12-01) or `:exvivo` (a different heart beating against a synthetic afterload, 2020-12-02).
The recording is split in four contiguous segments; `segments` picks which to concatenate.
The raw files store flow in L/min; it is converted here. Sampling period `h` = 5 ms.
"""
function load_pig(source::Symbol; segments=1:4)
    source in (:invivo, :exvivo) || throw(ArgumentError("source must be :invivo or :exvivo"))
    d = joinpath(DATADIR, "$(source)_data")
    col(f) = Float64.(readdlm(joinpath(d, f), ','; skipstart=2)[:, 2])
    p = reduce(vcat, [col("Ao_baseline1_AoP$i.csv") for i in segments])
    q = reduce(vcat, [col("Ao_baseline1_AoQ$i.csv") for i in segments])
    return p, Lmin_to_mLs.(q), 0.005
end

end
