# Figures for the README, written to docs/. Run from the repository root:
#   julia --project=examples examples/readme_figures.jl
include("idanalysis.jl")
docs = joinpath(dirname(@__DIR__), "docs")
mkpath(docs)
ink, series = "#2b2b28", ["#2a78d6", "#eb6834", "#1baf7a"] # validated categorical slots 1-3
default(fontfamily="sans-serif", grid=true, gridalpha=0.15, framestyle=:box, foreground_color_axis=:gray40,
    legend_foreground_color=nothing, lw=2, dpi=150)

# 1. What the package does: fits of the three models to one human beat
P, Q, h = load_stergiopulos1999()
smooth!(x, n=5) = append!(x, (collect(1:n-1) .* (x[1] - x[end]) / n) .+ x[end])
y, u = repeat(smooth!(copy(P)), 20), repeat(smooth!(copy(Q)), 20)
θs = Dict(G => [0.816, 1.238, 0.0571, 0.00571], G3 => [0.78, 1.79, 0.0349], G2 => [0.814, 1.66]) # Table 2 fits
t = h .* (0:length(P)-1)
pp = plot(t, P, color=ink, lw=3, label="Measured", ylabel="Pressure [mmHg]", xformatter=_ -> "", legend=:topright)
for ((G_, name, ls), c) in zip(((G, "4-element", :solid), (G3, "3-element", :dash), (G2, "2-element", :dot)), series)
    phat, _ = identify(θs[G_], solvex0(G_(θs[G_], h), u), u, y, h; θid=ones(Int, length(θs[G_])), x0id=false, x0solve=true, G_=G_)
    yhat = simulate(phat, solvex0(G_(phat, h), u), u, h; G_=G_)
    plot!(pp, t, yhat[1:length(P)], color=c, ls=ls, label=name)
end
pq = plot(t, Q, color=ink, label=:none, ylabel="Flow [mL/s]", xlabel="Time [s]")
savefig(plot(pp, pq, layout=grid(2, 1, heights=[0.68, 0.32]), size=(800, 520), left_margin=3Plots.mm,
    plot_title="Windkessel fits to a human aortic beat (Stergiopulos et al. 1999)", plot_titlefontsize=11),
    joinpath(docs, "fit.png"))

# 2. Why identifiability matters: profile of the fit over the inertance L. For each fixed L the other
# parameters are refitted; a flat profile means the data cannot determine L.
data = ((name="Human (Stergiopulos 1999)", y=y, u=u, h=h, θ=θs[G]),
        (name="Pig in vivo", y=load_pig(:invivo)[1], u=load_pig(:invivo)[2], h=0.005, θ=[3.77, 1.42, 0.0948, 0.511]))
Ls = exp10.(range(-3, 3, length=97))
plt = plot(xaxis=:log10, xlabel="Inertance L, fixed [mmHg s²/mL]", ylabel="MSE increase over best fit [%]",
    title="Cost profile over L, other parameters refitted", titlefontsize=11, legend=:right,
    size=(800, 400), left_margin=3Plots.mm, bottom_margin=3Plots.mm, ylims=(-2, 60))
for (d, c, ls) in zip(data, series, (:solid, :dash))
    mse = map(Ls) do L
        θ = [d.θ[1:3]; L]
        phat, res = identify(θ, solvex0(G(θ, d.h), d.u), d.u, d.y, d.h; θid=[1, 1, 1, 0], x0id=false, x0solve=true)
        res.minimum / length(d.u)
    end
    plot!(plt, Ls, 100 .* (mse ./ minimum(mse) .- 1), color=c, ls=ls, label=d.name)
end
savefig(plt, joinpath(docs, "identifiability.png"))
