# Multiple initialization, Hessian-based sensitivity summaries and plots used by the example scripts.
# Units: pressure mmHg, flow mL/s, time s. Plots can show flow in L/min with `flowunit=:Lmin`.
using WindkesselIdentification, ControlSystemsBase, ForwardDiff, LinearAlgebra, Random, Plots, Printf
using WindkesselIdentification: p, deal

flowscale(flowunit) = flowunit === :Lmin ? (mLs_to_Lmin(1.0), "L/min") : (1.0, "mL/s")
"Volume [mL] through the aorta, integrated from flow `q` [mL/s]."
volume(q, h) = h .* cumsum(q)
θlabels(n) = ["Rp", "C", "Rc", "L"][1:n]

"""
Simulated pressure for (θhat, x0hat) against the measured `y`, with the flow `u` below.
"""
function plot_result(θhat, x0hat, y, u, h; G_=G, flowunit=:mLs)
    yhat, xhat = simulate(θhat, x0hat, u, h; xout=true, G_=G_)
    k, unit = flowscale(flowunit)
    t = range(0, step=h, length=length(u))
    plty = plot(t, [y yhat], xticks=:none, label=["Experiment" "Model"], ylabel="Pressure [mmHg]",
        title=join((@sprintf("%s=%.3g", l, v) for (l, v) in zip(θlabels(length(θhat)), θhat)), ", "))
    pltu = plot(t, k .* u, label=:none, ylabel="Flow [$unit]", xlabel="Time [s]")
    return yhat, xhat, t, plot(plty, pltu, grid=true, layout=grid(2, 1, heights=[0.67, 0.33]))
end

"""
Fit from `n` random initial points: Rp (if `paramdist[1]` is false) stays at θ0, the others are drawn
uniformly on [0.001, 10] times θ0. The initial state is solved from periodicity at every step.
Returns the initial points, estimates and Optim results.
"""
function id_multiple_init(u, y, θ0, n=100; h=0.005, G_=G, paramdist=[false, true, true, true],
        σy=0, σu=0, rng=Xoshiro(0))
    θid = ones(Int, length(θ0))
    θ0s, θhats, x0hats, res = [], [], [], []
    for _ in 1:n
        y_in = y .+ σy .* randn(rng, length(y))
        u_in = u .+ σu .* randn(rng, length(u))
        θ0_ = [d ? θ * (0.001 + 9.999 * rand(rng)) : θ for (d, θ) in zip(paramdist, θ0)]
        x00_ = solvex0(G_(θ0_, h), u_in)
        phat, res_ = identify(θ0_, x00_, u_in, y_in, h; θid=θid, x0id=false, x0solve=true, G_=G_)
        θhat_, _ = deal(phat, IdConditions(θ0_, x00_, θid, false, true, G_))
        push!(θ0s, θ0_); push!(θhats, θhat_); push!(x0hats, solvex0(G_(θhat_, h), u_in)); push!(res, res_)
    end
    return θ0s, θhats, x0hats, res
end

"""
Summarise multiple initialization: one NamedTuple per fit, sorted by MSE, with the Hessian of the
cost at the estimate (and its condition number and SVD). Failed (NaN) fits are dropped.
"""
function multiinitresults(θhats, x0hats, u, y, h; G_=G)
    rows = map(θhats, x0hats) do θ, x0
        id = IdConditions(θ, x0, ones(Int, length(θ)), false, true, G_)
        f(p) = cost(p, id, u, y, h)
        H = ForwardDiff.hessian(f, θ)
        (θ=θ, x0=x0, MSE=f(θ) / length(u), hess=H, hesscond=cond(H), hessSVD=svd(H))
    end
    ok = filter(r -> !isnan(r.MSE), rows)
    nanratio = 1 - length(ok) / length(rows)
    return sort(ok, by=r -> (r.MSE, r.hesscond)), nanratio
end

"""
Histograms of the initial (θ0) and fitted (θhat) parameters from multiple initialization.
"""
function fithistograms(θ0s, θhats)
    labels = θlabels(length(θ0s[1]))
    hists0 = [histogram([θ[i] for θ in θ0s], bins=100, ylabel=labels[i]) for i in eachindex(labels)]
    hists = [histogram([θ[i] for θ in θhats], bins=100) for i in eachindex(labels)]
    title!(hists0[1], "initial θ"); title!(hists[1], "fit θ")
    plot(plot(hists0..., layout=(length(labels), 1)), plot(hists..., layout=(length(labels), 1)),
        layout=(1, 2), legend=:none)
end

"Write `cols` (a NamedTuple of equal-length vectors) as CSV with `#` comment lines, for pgfplots."
function writecsv(file, comment, cols)
    mkpath(dirname(file))
    open(file, "w") do io
        foreach(l -> println(io, "# ", l), split(comment, '\n'))
        println(io, join(keys(cols), ","))
        foreach(r -> println(io, join(r, ",")), zip(cols...))
    end
end

"""
The paper's procedure: multiple initialization of the 4-element model (Rp fixed at mean(y)/mean(u),
the others drawn around the v1.0 start [Rp, 1, 1, 1] in L/min units), then the 3- and 2-element models
started from the best 4-element estimate. Returns the sorted multi-init summaries per model.
"""
function fit_models(u, y, h; n=100, rng=Xoshiro(0))
    Rp = sum(y) / sum(u)
    θ0 = [Rp, 1000 / 60, 0.06, 0.06] # [Rp, 1, 1, 1] in the v1.0 L/min units
    fits = Dict{Any,Any}()
    for (G_, m) in ((G, 4), (G3, 3), (G2, 2))
        start = G_ === G ? θ0 : fits[G][1].θ[1:m]
        θ0s, θhats, x0hats, _ = id_multiple_init(u, y, start, n; h=h, G_=G_, paramdist=[false; trues(m - 1)], rng=rng)
        fits[G_], nanratio = multiinitresults(θhats, x0hats, u, y, h; G_=G_)
        best = fits[G_][1]
        @printf("WK%d: θ = %s, MSE = %.3g, cond(H) = %.3g, failed fits %.0f%%\n", m,
            string(round.(best.θ, sigdigits=3)), best.MSE, best.hesscond, 100nanratio)
        G_ === G && display(fithistograms(θ0s, θhats))
    end
    return fits
end

"Bode magnitude and phase of the continuous-time models in `fits`, at frequencies `w` [rad/s]."
function bodeplot_fits(fits, h, w=exp10.(range(-3, 3, length=500)); title="")
    pm, pp = plot(xaxis=:log, yaxis=:log, ylabel="|Z| [mmHg s/mL]", title=title), plot(xaxis=:log, ylabel="Phase [deg]", xlabel="ω [rad/s]")
    for (G_, ls, m) in ((G, :solid, 4), (G3, :dash, 3), (G2, :dot, 2))
        mag, phase, _ = bode(G_(fits[G_][1].θ, h; discretize=false), w)
        plot!(pm, w, vec(mag), ls=ls, label="WK$m"); plot!(pp, w, vec(phase), ls=ls, label=:none)
    end
    plot(pm, pp, layout=(2, 1))
end
