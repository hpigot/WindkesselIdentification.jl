"""
Identification of 2-, 3- and parallel 4-element Windkessel models from measured aortic pressure and flow.

Units throughout: pressure in mmHg, flow in mL/s, time in s, so that
Rp, Rc [mmHg s/mL], C [mL/mmHg], L [mmHg s^2/mL].
"""
module WindkesselIdentification

using LinearAlgebra, DelimitedFiles, ControlSystemsBase, Optim, LineSearches, ForwardDiff

export IdConditions, identify, identifyglobal, cost, simulate, G, G3, G2, p, deal, solvex0,
    params_stergiopulos1999total, mLs_to_Lmin, Lmin_to_mLs, load_stergiopulos1999, load_pig

"L/min per mL/s"
mLs_to_Lmin(q) = q * 60 / 1000
"mL/s per L/min"
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
Generates parallel 4-element Windkessel instance with parameter values of θ=[Rp,C,Rc,L], ZOH discretized with period h.
"""
G(θ, h; discretize=true) = statespace(wk4(θ), h, discretize)

"""
Generates 3-element Windkessel instance with parameter values of θ=[Rp,C,Rc], ZOH discretized with period h. Corresponds to 4 element with L = Inf.
"""
G3(θ, h; discretize=true) = statespace(wk3(θ), h, discretize)

"""
Generates 2-element Windkessel instance with parameter values of θ=[Rp,C], ZOH discretized with period h. Corresponds to 4 element with Rc = 0.
"""
G2(θ, h; discretize=true) = statespace(wk2(θ), h, discretize)

# Map a model constructor (G, G3, G2) to its diagonal form, so `IdConditions.G_` keeps working.
ctmodel(::typeof(G), θ) = wk4(θ)
ctmodel(::typeof(G3), θ) = wk3(θ)
ctmodel(::typeof(G2), θ) = wk2(θ)

"Initial state as a tuple of the model's length and of the type the recursion will produce (type-stable loops)."
function initstate((ad, bd, c, d)::NTuple{4,Any}, x0, u)
    T = promote_type(eltype(ad), eltype(bd), eltype(c), typeof(d), eltype(x0), eltype(u))
    return ntuple(i -> T(x0[i]), Val(length(ad)))
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
    solvex0(sys, u)

Initial state for which the response to `u` is periodic: x(n) = x(0), with n = length(u).
With s the state reached from x(0) = 0 after n steps, x(0) = (I - A^n)^-1 s.
`sys` is a discrete StateSpace, e.g. `G(θ, h)`.
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
Returns the cost (to be minimized).
"""
function cost(p,id::IdConditions,u,y,h)
    θ,x0 = deal(p,id)
    md = zoh(ctmodel(id.G_, θ), h)
    if id.x0solve x0 = solvex0(md,u) end
    return sse(md, u, y, x0)
end

"""
Simulate the 4-element model G(θ,h) (or `G_`) with input vector u, starting in initial condition x0.
"""
function simulate(θ,x0,u,h; xout=false, G_=G)
    md = zoh(ctmodel(G_, θ), h)
    y = similar(u, promote_type(eltype(u), eltype(θ)))
    X = xout ? similar(y, length(u), length(x0)) : nothing
    sim!(y, md, u, x0; X=X)
    return xout ? (y, X) : y
end

"""
Identifies subset of {system parameters θ, initial state x0}.
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
        lp = vcat(zeros(+(id.θid...)),fill(-Inf,length(x00)))
        dfc = TwiceDifferentiableConstraints(lp, up)
        res = optimize(df,dfc,p0,IPNewton(),opts)
    else
        res = optimize(df,p0,Newton(alphaguess = LineSearches.InitialPrevious(), linesearch = LineSearches.BackTracking()),opts)
    end

    phat = Optim.minimizer(res)
    return phat,res
end

"""
Identifies subset of {system parameters θ, initial state x0} using NelderMead.
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
Build parameter vector for optimization
"""
function p(θ,x0,id::IdConditions)
    p = θ[findall(Bool.(id.θid))]
    if id.x0id append!(p,x0) end
    return Float64.(p)
end

"""
Retrieve original parametrization.
"""
function deal(p,id::IdConditions)
    pidx = cumsum(id.θid)
    θ = [Bool(id_) ? p_ : θ0_ for (p_,θ0_,id_) in zip(p[pidx],id.θ0,id.θid)]
    x0 = id.x0id ? p[sum(id.θid).+(1:length(id.x00))] : id.x00
    return θ,x0
end

"""
Parameters from stergiopulos1999total (Table 1, type A beat).
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

Aortic pressure `p` [mmHg] and flow `q` [mL/s] from a 70 kg pig, `source` = `:invivo`
(2020-12-01) or `:exvivo` (heart beating against a synthetic afterload, 2020-12-02).
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
