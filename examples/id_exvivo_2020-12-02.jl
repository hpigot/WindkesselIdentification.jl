# using DataFrames, Dates, Plots, StatsPlots, CSV, Distributions
using CSV
include("oeid.jl");
include("idanalysis.jl");

# import and plot pressure (P) and flow (Q) waveforms
AoP, AoQ = importPQ("exvivo_data/Ao_baseline1_Ao",4);
checksampling(["AoP", "AoQ"], AoP, AoQ)
pltPQ = plotPQ(AoP, AoQ, "ex vivo")

h = 0.005 # sampling rate

title = "ex vivo"
for i = 1:4
    pltPQ = plot(plot(AoP[i].timestamp, AoP[i].vb_va, xticks=:none, ylabel = "AoQ [L/min]"), plot(AoQ[i].timestamp, AoQ[i].vb_va, ylabel = "AoQ [L/min]"), xlims = (AoQ[i][1,:timestamp]|>Dates.value, AoQ[i][end,:timestamp]|>Dates.value), layout = (2,1));
    V, pltPV = pvloop(AoP[i].vb_va,AoQ[i].vb_va,h)

    display(plot(pltPV, pltPQ,  layout = (1, 2), size = (1200,400), legend=:none, title = [title*" $(i) at ~125 bpm" "" ""]))
    # savefig("./results/pv_exvivo$(i)_" * Dates.format(now(), "yyyymmdd") * ".png")
end

# id first attempt
u = vcat([AoQ[i].vb_va for i in 1:4]...);
y = vcat([AoP[i].vb_va for i in 1:4]...);
Rp = mean(y)/mean(u);
θ0 = [Rp,1, 1, 1]
x00 = solvex0(G(θ0,h), u)
θid = [1,1,1,1]
x0id = false
x0solve = true
idexvivo = IdConditions(θ0,x00,θid,x0id,x0solve)
@time phat, res = identify(θ0,x00,u,y,h,θid=θid,x0id=x0id,x0solve=x0solve);
θhat, x0hat = deal(phat, idexvivo)
θ0, x00
x0hatsolve = solvex0(G(θhat,h),u)
hess = Optim.trace(res)[end].metadata["h(x)"]
svd(hess).U

# check result
yhat, xhat, t, plt = plot_result(θhat,x0hatsolve,y,u,h)
plt

# multiple initialization, identify 4 element windkessel model
θ0s, θhats, x0hats, res = id_multiple_init(u, y, x00, θ0, 100, θid=θid,x0id=x0id,x0solve=x0solve, paramdist=[false,true,true,true]);

# summarize results
results4, nanratio = multiinitresults(θhats, x0hats, res);
results4[1,:hess]
results4[1,:hessSVD].U
bestexvivo4 = results4[1,[1:7...,11,12]]

θbest4 = Array(results4[1, 1:length(θ0)]);
[θ0, round.(θbest4, sigdigits=3)]
x0best4 = Array(results4[1,["x01", "x02"]]);
[x00, round.(x0best4, sigdigits=3)]
yhat4, xhat4, t, plt_best = plot_result(θbest4, x0best4, y, u, h)
plot(plt_best, size=(1200,400))

# savefig("results/best_fit_exvivo4_$(Dates.format(now(), "yyyymmdd")).png")
# CSV.write("results/results_exvivo4_$(Dates.format(now(), "yyyymmdd")).csv",results4)
plot(t, xhat4, title = "exvivo $(round.(θbest4, sigdigits=3))", xlabel = "Time [s]", label=["x1 (pk-pk ≈ $(round(maximum(xhat4[1:100,1])-minimum(xhat4[1:100,1]), sigdigits=2)))" "x2"], size=(1200,400))

# plot parameter histograms before and after fit
θlabels = ["Rp" "C" "Rc" "L"]
fithistograms(θ0s,θhats,θlabels)
# savefig("./results/param_conv_" * Dates.format(now(), "yyyymmdd-HHMMSS") * ".png")

w = exp10.(range(-3, stop=3, length=500))
mag4, phase4, w = bode(G(θbest4,h,discretize=false),w)
magplt = plot(w,mag4[:],xaxis=:log, yaxis=:log, label="WK4",title="ex vivo");
phaseplt = plot(w,phase4[:], xaxis=:log, label="WK4");
plot(magplt, phaseplt, layout=(2,1))

# multiple initialization, identify 3 element windkessel model
θ0 = copy(θbest4[1:3])
x00 = solvex0(G3(θ0,h), u)
θid = [1,1,1]
x0id = false
x0solve = true
θ0s, θhats, x0hats, res = id_multiple_init(u, y, x00, θ0, 100, θid=θid,x0id=x0id,x0solve=x0solve, G_=G3, paramdist=[false,true,true]);

results3, nanratio = multiinitresults(θhats, x0hats, res);
θbest3 = Array(results3[1,1:3])
x0best3 = [results3[1,"x01"]]
results3[1,:hess]
results3[1,:hessSVD].U
bestexvivo3 = results3[1,["Rp", "C", "Rc", "x01", "MSE","hesscond"]]
# CSV.write("results/results_exvivo3_$(Dates.format(now(), "yyyymmdd")).csv",results3)
# results3 = CSV.read("results/results_exvivo3_20210308.csv", DataFrame)

yhat3, = lsim(G3(θbest3,h),u,h.*(1:length(u)),x0best3)
mag3, phase3, w = bode(G3(results3[1,1:3],h,discretize=false),w)
gr()
mag3plt = plot(w,mag3[:],xaxis=:log, yaxis=:log, label="WK3",title="ex vivo");
phase3plt = plot(w,phase3[:], xaxis=:log, label="WK3");
plot(mag3plt, phase3plt, layout=(2,1))

# multiple initialization, identify 2 element windkessel model
θ0 = copy(θbest4[1:2])
x00 = solvex0(G2(θ0,h), u)
θid = [1,1]
θ0s, θhats, x0hats, res = id_multiple_init(u, y, x00, θ0, 100, θid=θid,x0id=x0id,x0solve=x0solve, G_=G2, paramdist=[false,true]);

results2, nanratio = multiinitresults(θhats, x0hats, res);
θbest2 = Array(results2[1,1:2])
x0best2 = [results2[1,"x01"]]
results2[1,:hess]
results2[1,:hessSVD].U
bestexvivo2 = results2[1,["Rp", "C", "x01", "MSE","hesscond"]]
CSV.write("results/results_exvivo2_$(Dates.format(now(), "yyyymmdd")).csv",results2)
# results2 = CSV.read("results/results_exvivo2_20210308", DataFrame)

yhat2, = lsim(G2(θbest2,h),u,h.*(1:length(u)),x0best2)
mag2, phase2, w = bode(G2(θbest2,h,discretize=false),w)
gr()
mag2plt = plot(w,mag2[:],xaxis=:log, yaxis=:log, label="WK2",title="ex vivo");
phase2plt = plot(w,phase2[:], xaxis=:log, label="WK2");
plot(mag2plt, phase2plt, layout=(2,1))

# load results
# results4 = CSV.read("results/results_exvivo4_20210309.csv", DataFrame)
# θbest4 = Array(results[1,1:4])
# x0best4 = solvex0(G(θbest4,h),u)
# yhat4 = simulate(θbest4,x0best4,u,h)[:,1]
v, = pvloop(y,u,h)
plot([y,yhat4],xlims=(0,700))
plot([v[450:600],v[450:600]],[y[450:600],yhat[450:600]])

# export bode data for pgfplots
comment= *(
    "# fit to data: ex vivo Ao_baseline1 20201202\n",
    "# mag4,phase4: 4-element parallel Windkessel model\n",
    "# θhat = [Rc,L,C,Rp]\n",
    "# θhat = $(θbest4)\n",
    "# mag3,phase3: 3-element parallel Windkessel model\n",
    "# θhat = [Rc,C,Rp]\n",
    "# θhat = $(θbest3)\n",
    "# mag2,phase2: 2-element parallel Windkessel model\n",
    "# θhat = [C,Rp]\n",
    "# θhat = $(θbest2)\n"
    )
filename = "bodeexvivo.csv"
open(filename, "w") do f
    write(f, comment)
end
CSV.write(filename, DataFrame(w = w,
mag4 = mag4[:], phase4 = phase4[:],
mag3 = mag3[:], phase3 = phase3[:],
mag2 = mag2[:], phase2 = phase2[:]),header=true,append=true)

# export data for pgfplots
beatstart = 582
beatend = 679
comment = *("# data: ex vivo Ao_baseline1 20201202, 7th beat only $(beatstart):$(beatend)\n",
"# t: time in seconds\n",
"# q: measured Aortic flow in L/min\n",
"# p: measured Aortic pressure in mmHg\n",
"# phat4: modelled Aortic pressure in mmHg\n",
"# v: measured volume through the aorta in ml\n",
"# epsilon4: error signal of the model p-phat4\n",
"# 4-element parallel Windkessel model\n",
"# θhat = [Rc,L,C,Rp]\n",
"# θhat = $(θbest4); x0hat = $(x0best4)\n",
"# as well as the corresonding phat and epsilon values for\n",
"# 3-element parallel Windkessel model\n",
"# θhat = [Rc,C,Rp]\n",
"# θhat = $(θbest3); x0hat = $(x0best3)\n",
"# 2-element parallel Windkessel model\n",
"# θhat = [C,Rp]\n",
"# θhat = $(θbest2); x0hat = $(x0best2)\n")
filename = "pqvexvivo.csv"
open(filename, "w") do f
    write(f, comment)
end
CSV.write(filename, DataFrame(t = round.(collect(0:beatend-beatstart)*h, digits=5), q = u[beatstart:beatend], p = y[beatstart:beatend], phat4 = yhat4[beatstart:beatend], phat3 = yhat3[beatstart:beatend], phat2 = yhat2[beatstart:beatend], v = v[beatstart:beatend].-v[beatstart], epsilon4 = y[beatstart:beatend].-yhat4[beatstart:beatend],epsilon3 = y[beatstart:beatend].-yhat3[beatstart:beatend],epsilon2 = y[beatstart:beatend].-yhat2[beatstart:beatend]),header=true,append=true)
