using CSV
include("oeid.jl");
include("idanalysis.jl");

prefix = "stergiopulos1999_data/"
AoP = CSV.read(prefix*"apressure.csv", DataFrame, header=false, types=[Float64, Float64])
AoQ = CSV.read(prefix*"aflow.csv", DataFrame, header=false, types=[Float64, Float64])
rename!(AoP, ["timestamp", "vb_va"])
rename!(AoQ, ["timestamp", "vb_va"])
size(AoP) == size(AoQ) || @warn "Pressure and flow waveforms not equal in size."
AoQ[!,"vb_va"] = AoQ[!,"vb_va"].*(60/1000) # convert from mL/s to L/min
pltPQ = plotPQ([AoP], [AoQ], "stergiopulos1999", size=(800,400))

h = 0.005 # sampling rate

# input beats
# FOH fill to match initial and final value of beat sequence
smooth!(x,n=5) = append!(x,(collect(1:n-1).*(x[1]-x[end])/n).+x[end])
y = repeat(smooth!(AoP[:,:vb_va]),20)
u = repeat(smooth!(AoQ[:,:vb_va]),20)

# with parameters from paper
# θ = Rp,C,Rc,L
θ1999mLs, _ = params_stergiopulos1999total()
mLs2Lmin(θ) = Diagonal([1000/60, 60/1000, 1000/60, 1000/60])*θ
θ1999 = mLs2Lmin(θ1999mLs) # convert impedance units to match mmHg pressures and L/min flows
x01999 = solvex0(G(θ1999,h),u)
y1999, x1999, t, plt = plot_result(θ1999,x01999,y,u,h)
plt

# first ID attempt
Rp = mean(y)/mean(u);
θ0 = [Rp, 1, 1, 1]
x00 = solvex0(G(θ0,h), u)
θid = [1,1,1,1]
x0id = false
x0solve = true
id1999 = IdConditions(θ0,x00,θid,x0id,x0solve)
@time phat, res = identify(θ0,x00,u,y,h,θid=θid,x0id=x0id,x0solve=x0solve);
θhat, x0hat = deal(phat, id1999)
θ0, x00
x0hatsolve = solvex0(G(θhat,h),u)
hess = Optim.trace(res)[end].metadata["h(x)"]
svd(hess).U
_, _, _, plt = plot_result(θhat,x0hatsolve,y,u,h)
plot(plt, xlims=[0,1])

# check hessian and MSE with original parameters
@time phat1999,res1999 = identify(θ1999,solvex0(G(θ1999,h),u),u,y,h,θid=θid,x0id=x0id,x0solve=x0solve);
hess1999 = res1999.trace[1].metadata["h(x)"]
cond(hess1999)
svd(hess1999)

# multiple initialization, identify 4 element windkessel model
θ0s, θhats, x0hats, res = id_multiple_init(u, y, x00, θ0, 100, θid=θid,x0id=x0id,x0solve=x0solve, paramdist=[false,true,true,true]);

results4, nanratio = multiinitresults(θhats, x0hats, res);
results4[1,:hess]
results4[1,:hessSVD].U
best1999_4 = results4[1,[1:7...,11,12]]

θbest4 = convert(Array, results4[1, 1:length(θ0)]);
[θ0, round.(θbest4, sigdigits=3)]
x0best4 = Array(results4[1,["x01", "x02"]]);
[x00, round.(x0best4, sigdigits=3)]
_, _, t, plt_best = plot_result(θbest4, x0best4, y, u, h)
plot(plt_best, size=(1200,400))
# savefig("results/best_fit_1999_4_$(Dates.format(now(), "yyyymmdd")).png")
# CSV.write("results/results_1999_4_$(Dates.format(now(), "yyyymmdd")).csv",results4)

# plot parameter histograms before and after fit
θlabels = ["Rp" "C" "Rc" "L"]
fithistograms(θ0s,θhats,θlabels)
# savefig("./results/param_conv_" * Dates.format(now(), "yyyymmdd-HHMMSS") * ".png")

yhat4, = lsim(G(θbest4,h),u,h.*(1:length(u)),x0best4)
yhat4 = yhat4[:]
w = exp10.(range(-3, stop=3, length=500))
mag4, phase4, w = bode(G(θbest4,h,discretize=false),w)
mag4plt = plot(w,mag4[:],xaxis=:log, yaxis=:log, label="WK4",title="1999");
phase4plt = plot(w,phase4[:], xaxis=:log, label="WK4");
plot(mag4plt, phase4plt, layout=(2,1))

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
best1999_3 = results3[1,["Rp", "C", "Rc", "x01", "MSE","hesscond"]]
# CSV.write("results/results_1999_3_$(Dates.format(now(), "yyyymmdd")).csv",results3)
# results3 = CSV.read("results/results_1999_3_20210309.csv", DataFrame)

yhat3, = lsim(G3(θbest3,h),u,h.*(1:length(u)),x0best3)
yhat3 = yhat3[:]
mag3, phase3, w = bode(G3(results3[1,1:3],h,discretize=false),w)
gr()
mag3plt = plot(w,mag3[:],xaxis=:log, yaxis=:log, label="WK3",title="1999");
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
best1999_2 = results2[1,["Rp", "C", "x01", "MSE","hesscond"]]
# CSV.write("results/results_1999_2_$(Dates.format(now(), "yyyymmdd")).csv",results2)
# results2 = CSV.read("results/results_1999_2_20210309", DataFrame)

yhat2, = lsim(G2(θbest2,h),u,h.*(1:length(u)),x0best2)
yhat2 = yhat2[:]
mag2, phase2, w = bode(G2(θbest2,h,discretize=false),w)
gr()
mag2plt = plot(w,mag2[:],xaxis=:log, yaxis=:log, label="WK2",title="1999");
phase2plt = plot(w,phase2[:], xaxis=:log, label="WK2");
plot(mag2plt, phase2plt, layout=(2,1))

y = AoP.vb_va
u = AoQ.vb_va
v, = pvloop(y,u,h)
y1999 = y1999[1:length(y),1]
# load results
# results4 = CSV.read("results/results_1999_4_20210309.csv", DataFrame)
# svd1999 = eval(Meta.parse(results4[1,:hessSVD]))
# x0best4 = solvex0(G(θbest4,h),u)
# θbest4 = Array(results4[1,1:4])
# yhat4 = simulate(θbest4,x0best4,u,h)[:,1]
svd1999 = results4[1,:hessSVD]
plot([y,y1999,yhat4],xlims=(0,300))
plot([v,v,v],[y,y1999,yhat4])

# export bode data for pgfplots
comment= *(
    "# fit to data: published data stergiopulos1999total doi = {10.1152/ajpheart.1999.276.1.H81}, Figure 4A type A beat.\n",
    "# mag4,phase4: 4-element parallel Windkessel model\n",
    "# θhat = [Rc,L,C,Rp]\n",
    "# θhat = $(θbest4)\n",
    "# mag3,phase3: 3-element parallel Windkessel model\n",
    "# θhat = [Rc,C,Rp]\n",
    "# θhat = $(θbest3)\n",
    "# mag2,phase2: 2-element parallel Windkessel model\n",
    "# θhat = [C,Rp]\n",
    "# θhat = $(θbest2)\n")
filename = "bodesterg1999.csv"
open(filename, "w") do f
    write(f, comment)
end
CSV.write(filename, DataFrame(w = w,
mag4 = mag4[:], phase4 = phase4[:],
mag3 = mag3[:], phase3 = phase3[:],
mag2 = mag2[:], phase2 = phase2[:]),header=true,append=true)

# export data for pgfplots
beatstart = 1
beatend = length(AoP.vb_va)
comment = *("# data: from published data stergiopulos1999total doi = {10.1152/ajpheart.1999.276.1.H81}, Figure 4A type A beat.\n",
"# t: time in seconds\n",
"# q: published measured Aortic flow in L/min, parsed using https://automeris.io/WebPlotDigitizer\n",
"# p: published measured Aortic pressure in mmHg, parsed using https://automeris.io/WebPlotDigitizer\n",
"# phat1999: modelled Aortic pressure in mmHg, using parameters published in paper θhat1999\n",
"# phat4: modelled Aortic pressure in mmHg\n",
"# v: measured volume through the aorta in ml\n",
"# epsilon4: error signal of the model p-phat4\n",
"# 4-element parallel Windkessel model\n",
"# θhat = [Rp,C,Rc,L]\n",
"# θhat1999 = $(θ1999); solved for x0hat1 = $(x01999)\n",
"# θhat = $(θbest4); x0hat2 = $(x0best4)\n",
"# as well as the corresonding phat and epsilon values for\n",
"# 3-element parallel Windkessel model\n",
"# θhat = [Rc,C,Rp]\n",
"# θhat = $(θbest3); x0hat = $(x0best3)\n",
"# 2-element parallel Windkessel model\n",
"# θhat = [C,Rp]\n",
"# θhat = $(θbest2); x0hat = $(x0best2)\n")
filename = "pqvsterg1999.csv"
open(filename, "w") do f
    write(f, comment)
end
CSV.write(filename, DataFrame(t = round.(collect(0:beatend-beatstart)*h, digits=5), q = u[beatstart:beatend],
p = y[beatstart:beatend],
phat1999 = y1999[beatstart:beatend],
phat4 = yhat4[beatstart:beatend],
phat3 = yhat3[beatstart:beatend],
phat2 = yhat2[beatstart:beatend],
v = v[beatstart:beatend].-v[beatstart],
epsilon1999 = y[beatstart:beatend].-y1999[beatstart:beatend],
epsilon4 = y[beatstart:beatend].-yhat4[beatstart:beatend],
epsilon3 = y[beatstart:beatend].-yhat3[beatstart:beatend],
epsilon2 = y[beatstart:beatend].-yhat2[beatstart:beatend]),
header=true,append=true)
