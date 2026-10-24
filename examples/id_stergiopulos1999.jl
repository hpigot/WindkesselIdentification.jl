# Identify 4-, 3- and 2-element Windkessel models from the human type A beat of Stergiopulos et al. (1999),
# Fig. 4A (Table 2 and Figs. 2, 5, 6 of Pigot et al. 2021). Run from the repository root:
#   julia --project=examples examples/id_stergiopulos1999.jl
include("idanalysis.jl")
outdir = joinpath(@__DIR__, "output")

P, Q, h = load_stergiopulos1999()
# 20 repeated beats; a first-order hold over 5 samples closes the gap between the last and first sample
smooth!(x, n=5) = append!(x, (collect(1:n-1) .* (x[1] - x[end]) / n) .+ x[end])
y, u = repeat(smooth!(copy(P)), 20), repeat(smooth!(copy(Q)), 20)

# the parameters published by Stergiopulos et al.
θ1999, _ = params_stergiopulos1999total()
y1999, _, _, plt = plot_result(θ1999, solvex0(G(θ1999, h), u), y, u, h; flowunit=:Lmin)
display(plt)

fits = fit_models(u, y, h)
best = Dict(G_ => fits[G_][1] for G_ in (G, G3, G2))
yhat = Dict(G_ => simulate(best[G_].θ, best[G_].x0, u, h; G_=G_) for G_ in (G, G3, G2))
display(plot_result(best[G].θ, best[G].x0, y, u, h)[4])
display(bodeplot_fits(fits, h; title="Stergiopulos 1999"))
display(plot(volume(Q, h), [P y1999[1:length(P)] yhat[G][1:length(P)]], label=["y" "Stergiopulos 1999" "WK4"],
    xlabel="Volume [mL]", ylabel="Pressure [mmHg]"))

# export one beat for pgfplots
n = length(P)
comment = """data: Stergiopulos et al. (1999) doi:10.1152/ajpheart.1999.276.1.H81, Fig. 4A type A beat, digitised at 200 Hz
t [s]; q flow [mL/s]; p, phat* pressure [mmHg]; v volume [mL]; epsilon* = p - phat*
theta = [Rp, C, Rc, L] in mmHg, mL/s, s
theta1999 = $(θ1999)
theta4 = $(best[G].θ)
theta3 = $(best[G3].θ)
theta2 = $(best[G2].θ)"""
writecsv(joinpath(outdir, "pqvsterg1999.csv"), comment,
    (t=round.(h .* (0:n-1), digits=5), q=Q, p=P, phat1999=y1999[1:n], phat4=yhat[G][1:n], phat3=yhat[G3][1:n],
     phat2=yhat[G2][1:n], v=volume(Q, h), epsilon4=P .- yhat[G][1:n], epsilon3=P .- yhat[G3][1:n], epsilon2=P .- yhat[G2][1:n]))
