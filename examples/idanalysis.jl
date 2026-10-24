using DataFrames, Dates, Plots, StatsPlots, Distributions, LinearAlgebra, Printf

function importPQ(csvprefix::String,nfiles::Int)
    P = fill(DataFrame(),nfiles)
    Q = fill(DataFrame(),nfiles)
    for i in 1:nfiles
        P[i] = CSV.read(csvprefix*"P$(i).csv", DataFrame, comment="#", types=[Time, Float64], normalizenames=true)
        Q[i] = CSV.read(csvprefix*"Q$(i).csv", DataFrame, comment="#", types=[Time, Float64], normalizenames=true)
        println("start (P,Q): ($(P[i].timestamp[1]),$(Q[i].timestamp[1]))\tlength: ($(length(P[i].vb_va)),$(length(Q[i].vb_va)))")
        Millisecond(-5) < (P[i].timestamp[1]-Q[i].timestamp[1]) < Millisecond(5) || @warn "t0 differs by >5ms between P$(i) and Q$(i)"
        length(P[i].vb_va) == length(Q[i].vb_va) || @warn "P Q lengths not equal. Samples may be missing."
    end
    return P,Q
end

function checksampling(labels::Array{String,1}, datasets::Array{DataFrame,1}...; boxplt=false)
    length(labels) == length(datasets) || throw(ArgumentError("Length of labels does not match number of datasets."))
    rateplts = Dict();
    for (data,label) in zip(datasets,labels)
        δts = [data[i].timestamp[2:end].-data[i].timestamp[1:end-1] for i in 1:length(data)]
        hs = [unique(δt) for δt in δts]
        println(label*" set of sampling rates")
        display(hs)
        inrange = Millisecond(4) .<= vcat(hs...) .<= Millisecond(6)
        .&(inrange...) || @warn "sampling rate outside range 4-6 milliseconds"
        if boxplt
            rateplts[label] = boxplot([[δt|>Dates.value|>x->x/1000 for δt in δts_] for δts_ in δts], ylabel = "δt [μs]")
            xticks!(:none)
        else
            rateplts[label] = histogram([[δt|>Dates.value|>x->x/1000 for δt in δts_] for δts_ in δts], xlabel = "δt [μs]", ylabel = "n", bins=20, alpha=0.5)
        end
        title!(label)
    end
    plot(values(rateplts)..., size = (length(rateplts)*300,400))
end

function plotPQ(P::Array{DataFrame,1}, Q::Array{DataFrame,1}, title::String; scatterplt=true, size=(1200,400))
    pltP = plot();
    pltQ = plot();
    length(P) == length(Q) || @warn "unequal number of dataframes in P and Q."
    for i in 1:length(P)
        nrow(P[i]) == nrow(Q[i]) || @warn "unequal number of rows in P[$(i)] and Q[$(i)]."
        scatter!(pltP, P[i].timestamp, P[i].vb_va, label = "P$(i)", size = size)
        xlabel!("Time [s]")
        ylabel!("P [mmHg]")
        scatter!(pltQ, Q[i].timestamp, Q[i].vb_va, label = "Q$(i)", size = size)
        xlabel!("Time [s]")
        ylabel!("Q [L/min]")
    end
    plot(pltP,pltQ, layout=(2,1), title=[title ""], xlabel = ["" "Time [s]"])
end

"""
Takes time aligned vectors pressure `p` and flow `q` in L/min with samling rate h and returns volume vector `v` and a PV plot.
"""
function pvloop(p,q,h)
    #FIXME: peak detect and plot beat by beat.
    v = 1000/60*h.*cumsum(q) # L/min ⇒ milliliters
    pltPV = plot(v, p, ylabel="Pressure [mmhg]", xlabel="Volume [mL]", legend=:none);
    return v, pltPV
end

"""
Takes dataframes p and q (with columns "timestamp" and "vb_va") of pressure in mmHg and flow in L/min and returns a plot including p vs t, q vs t, and p vs volume.
"""
function plotPQV(p::DataFrame, q::DataFrame; h=0.005, pheight= 0.7, size=(800,400))
    nrow(p) == nrow(q) || @warn "unequal number of rows in P and Q."
    pltP = plot(p.timestamp, p.vb_va)
    ylabel!("Pressure [mmHg]")
    pltQ = plot(q.timestamp, q.vb_va)
    ylabel!("Flow [L/min]")
    pltPQ = plot(pltP,pltQ, layout = grid(2, 1, heights=[pheight,1-pheight]), xlabel = ["" "Time [s]"], legend=:none)
    v = 1000/60*h.*cumsum(q.vb_va) # L/min ⇒ milliliters
    pltPV = plot(v, p.vb_va, xlabel="Volume [mL]", legend=:none, ylabel="Pressure [mmHg]");
    plt = plot(pltPQ,pltPV,layout=(1,2),size=size,grid=true)
    return plt, v
end

function plot_result(θhat, x0hat, y, u, h)
    yhat, xhat = simulate(θhat,x0hat,u,h,xout=true)
    t = range(0,step=h,length=length(u))
    plty = plot([t,t],[y,yhat], xticks=:none, label = ["Experiment" "Model"])
    title!(@sprintf "Model Fit: Rp=%0.2f, C=%0.2f, Rc=%0.2f L=%0.2f (P̄/Q̄=%0.2f)" vcat(θhat, mean(y)/mean(u))...)
    ylabel!("AoP [mmHg]")
    pltu = plot(t,u,label=:none)
    ylabel!("AoQ [L/min]")
    xlabel!("Time [s]")
    yhat, xhat, t, plot(plty, pltu, grid=true, layout = grid(2, 1, heights=[0.67,0.33]))
end

function id_multiple_init(u, y, x00, θ0, n=100; h=0.005, θid=[1,1,1,1], x0id=true,x0solve=false, G_=G, paramdist=[true,true,true,true], σy=0, σu=0,)
    # fixme add check on paradist length
    id = IdConditions(θ0,x00,θid,x0id,x0solve,G_)
    
    θdists = fill(Uniform(0,1),length(θ0))
    for (i,dist) in enumerate(paramdist)
        if (dist)
            θdists[i] = Uniform(0.001*θ0[i],10*θ0[i])
        end
    end

    θ0s = []
    θhats = []
    x0hats = []
    res = []

    for _ in 1:n
        y_in = σy == 0 ? y : y + randn(length(u),1).*σy
        u_in = σu == 0 ? u : u + randn(length(u),1).*σu
        θ0_ = [dist ? rand(θdist) : θ0_ for (dist, θdist, θ0_) in zip(paramdist, θdists, θ0)]
        x00_ = solvex0(G_(θ0_,h), u_in)
        @time phat, res_ = identify(θ0_,x00_,u_in,y_in,h,θid=θid,x0id=x0id,x0solve=x0solve, G_=G_);
        θhat_, x0hat_ = deal(phat,id)
        if id.x0solve x0hat_ = solvex0(G_(θhat_,h),u_in) end

        push!(θ0s, θ0_)
        push!(θhats, θhat_)
        push!(x0hats, x0hat_)
        push!(res, res_)
    end
    θ0s, θhats, x0hats, res
end

"""
Summarize results from multiple initialization.
"""
function multiinitresults(θhats, x0hats, res)
    # fixme uses global u
    θlabels = ["Rp", "C", "Rc", "L"]
    # generalize for any θ:
    # θlabels = ["θ$(n)" for n in 1:length(θhats[1])]
    x0labels = ["x0$(n)" for n in 1:length(x0hats[1])]

    results = DataFrame([Symbol(label)=>Float64[] for label in vcat(θlabels[1:length(θhats[1])],x0labels)]...)

    for (θhat, x0hat) in zip(θhats, x0hats)
        push!(results, vcat(θhat, x0hat))
    end

    last_iters = [Optim.trace(res_)[end] for res_ in res]; # use other than end?
    results.MSE = [last_iter.value for last_iter in last_iters]./length(u);
    results.g = [last_iter.metadata["g(x)"] for last_iter in last_iters];
    results.hess = [last_iter.metadata["h(x)"] for last_iter in last_iters];
    results.poshess = [isposdef(last_iter.metadata["h(x)"]) for last_iter in last_iters];
    results.hesscond = [cond(last_iter.metadata["h(x)"]) for last_iter in last_iters];
    results.hessSVD = [svd(last_iter.metadata["h(x)"]) for last_iter in last_iters];
    results.idx = [1:length(θhats)...];

    @show sort!(results, [:MSE,:hesscond], rev=(false, false));

    nanratio = sum(isnan.(results[!,:MSE]))/length(results[!, :MSE])

    delete!(results, isnan.(results[!,:MSE]))
    
    return results, nanratio
end

"""
plot parameter histograms before (θ0) and after (θhat) multiple initialization fits.
"""
function fithistograms(θ0s,θhats,θlabels)
    hists0 = []
    hists = []
    for i in 1:length(θlabels)
        push!(hists0, histogram([θ0[i] for θ0 in θ0s], bins=100, ylabel=θlabels[i]))
        hist = histogram([θhat[i] for θhat in θhats], bins=100)
        #vline!(hist, [θ0[i]], c=:red, lw=2, ls=:dash, label=θlabels[i]*" true")
        push!(hists, hist)
    end

    title!(hists0[1],"initial θ");
    hist0_plt = plot(hists0..., layout=(length(hists0),1), legend=:none);
    title!(hists[1],"fit θ");
    hist_plt = plot(hists..., layout=(length(hists0),1), legend=:none);

    plot(hist0_plt, hist_plt, layout=(1,2))
end