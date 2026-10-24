# Identify 4-, 3- and 2-element Windkessel models from porcine aortic pressure and flow, measured in vivo
# (2020-12-01) or with a heart beating ex vivo against a synthetic afterload (2020-12-02)
# (Table 2 and Figs. 3-6 of Pigot et al. 2021). Run from the repository root:
#   julia --project=examples examples/id_pig.jl invivo    (or exvivo)
include("idanalysis.jl")
outdir = joinpath(@__DIR__, "output")
source = Symbol(get(ARGS, 1, "invivo"))

y, u, h = load_pig(source) # the four recorded segments, concatenated
display(plot(volume(u, h), y, xlabel="Volume [mL]", ylabel="Pressure [mmHg]", legend=:none, title="$source"))

fits = fit_models(u, y, h)
# The 4-element fits reach equal MSE over a wide range of L (L is practically unidentifiable),
# so the best-MSE estimate of L is arbitrary.
best = Dict(G_ => fits[G_][1] for G_ in (G, G3, G2))
yhat = Dict(G_ => simulate(best[G_].θ, best[G_].x0, u, h; G_=G_) for G_ in (G, G3, G2))
display(plot_result(best[G].θ, best[G].x0, y, u, h; flowunit=:Lmin)[4])
display(bodeplot_fits(fits, h; title="$source"))

# export one beat for pgfplots (the beats shown in the paper)
beat = source === :invivo ? (512:679) : (582:679)
comment = """data: $(source) porcine aortic pressure and flow, Ao_baseline1, samples $(beat)
t [s]; q flow [mL/s]; p, phat* pressure [mmHg]; v volume [mL]; epsilon* = p - phat*
theta = [Rp, C, Rc, L] in mmHg, mL/s, s
theta4 = $(best[G].θ)
theta3 = $(best[G3].θ)
theta2 = $(best[G2].θ)"""
v = volume(u, h)
writecsv(joinpath(outdir, "pqv$(source).csv"), comment,
    (t=round.(h .* (0:length(beat)-1), digits=5), q=u[beat], p=y[beat], phat4=yhat[G][beat], phat3=yhat[G3][beat],
     phat2=yhat[G2][beat], v=v[beat] .- v[beat[1]], epsilon4=y[beat] .- yhat[G][beat], epsilon3=y[beat] .- yhat[G3][beat],
     epsilon2=y[beat] .- yhat[G2][beat]))
