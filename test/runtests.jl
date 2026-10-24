using WindkesselIdentification, Test, ControlSystemsBase, ForwardDiff, LinearAlgebra, Random, Dates, DelimitedFiles, Aqua
using WindkesselIdentification: wk2, wk3, wk4, zoh, sse

const h = 0.005
# Table 2 of Pigot et al. (2021), doi:10.1016/j.ifacol.2021.10.307, in the v1.0 units:
# R [mmHg/(L/min)], C [(L/min) s/mmHg], L [mmHg s/(L/min)]. (The paper's Table 1 labels C and L
# with minutes; the code used seconds.) `tomLs` converts to the mmHg, mL/s, s units used here.
tomLs(θ) = [θ[1] * 0.06, θ[2] / 0.06, (θ[3:end] .* 0.06)...]
# (data, model, θ, MSE, which θ entries are identifiable enough to compare)
const table2 = [
    (:human, G, [13.6, 0.0743, 0.952, 0.0952], 5.88, 1:4),
    (:human, G3, [13.0, 0.108, 0.582], 8.48, 1:3),
    (:human, G2, [13.6, 0.0996], 48.2, 1:2),
    (:invivo, G, [62.9, 0.0855, 1.58, 8.52], 30.9, 1:3), # L is practically unidentifiable
    (:invivo, G3, [61.3, 0.0877, 1.58], 30.9, 1:3),
    (:invivo, G2, [62.9, 0.0853], 77.1, 1:2),
    (:exvivo, G, [148, 0.345, 0.615, 31.3], 7.16, 1:3),
    (:exvivo, G3, [148, 0.346, 0.615], 7.16, 1:3),
    (:exvivo, G2, [147, 0.353], 23.3, 1:1), # C sits in a flat valley (refit gives 0.363)
]
const θpaper4 = tomLs(table2[1][3])
const θpaper2 = tomLs(table2[3][3])

readdlm_time(f) = string.(readdlm(f, ','; skipstart=2)[:, 1])

smooth!(x, n=5) = append!(x, (collect(1:n-1) .* (x[1] - x[end]) / n) .+ x[end])
function stergiopulos_beats(nbeats=20)
    P, Q, _ = load_stergiopulos1999()
    return repeat(smooth!(copy(P)), nbeats), repeat(smooth!(copy(Q)), nbeats)
end

models = ((G2, θpaper2), (G3, θpaper4[1:3]), (G, θpaper4))

@testset "WindkesselIdentification" begin

@testset "data and units" begin
    @test Lmin_to_mLs(mLs_to_Lmin(123.4)) ≈ 123.4
    @test Lmin_to_mLs(6) ≈ 100
    P, Q, hs = load_stergiopulos1999()
    @test length(P) == length(Q) == 166 && hs == h
    @test sum(P) / sum(Q) ≈ 0.80 atol = 0.01 # ≈ published Rp = 0.79 mmHg s/mL
    for source in (:invivo, :exvivo)
        p, q, _ = load_pig(source)
        @test length(p) == length(q)
        raw = Float64.(readdlm(joinpath(pkgdir(WindkesselIdentification), "data", "$(source)_data", "Ao_baseline1_AoQ1.csv"), ','; skipstart=2)[:, 2])
        @test q[1:length(raw)] ≈ raw .* 1000 ./ 60 # stored in L/min, returned in mL/s
        # timestamps: 5 ms sampling, P and Q rows aligned to within the 1 ms resolution
        for i in 1:4
            ts(f) = Time.(readdlm_time(joinpath(pkgdir(WindkesselIdentification), "data", "$(source)_data", f)))
            tp, tq = ts("Ao_baseline1_AoP$i.csv"), ts("Ao_baseline1_AoQ$i.csv")
            @test Dates.value(tp[end] - tp[1]) / 1e6 / (length(tp) - 1) ≈ 5.0 atol = 0.01
            @test maximum(abs, Dates.value.(tp .- tq)) <= 2e6
        end
    end
end

@testset "discretisation matches ControlSystemsBase c2d" begin
    for (Gm, θ) in models
        a = Gm(θ, h)
        b = c2d(Gm(θ, h; discretize=false), h, :zoh)
        @test a.A ≈ b.A && a.B ≈ b.B && a.C ≈ b.C && a.D ≈ b.D
    end
end

@testset "simulate matches lsim" begin
    _, u = stergiopulos_beats(2)
    θ = θpaper4
    x0 = [100.0, -8.0]
    @test simulate(θ, x0, u, h) ≈ vec(lsim(G(θ, h), reshape(u, 1, :); x0=x0).y)
    y, X = simulate(θ, x0, u, h; xout=true)
    @test X[1, :] == x0 && size(X) == (length(u), 2)
end

@testset "analytic responses" begin
    # WK2 step response is exact under ZOH: p(t) = Rp(1 - exp(-t/(Rp C)))
    Rp, C = θpaper2
    t = h .* (0:999)
    @test simulate(θpaper2, [0.0], ones(1000), h; G_=G2) ≈ Rp .* (1 .- exp.(-t ./ (Rp * C)))
    # WK4 sinusoidal steady state: amplitude |Z(jω)| (ZOH half-sample delay is negligible at h = 5 ms)
    Rp, C, Rc, L = θpaper4
    ω = 2π * 1.2
    Z(s) = s * L * Rc / (s * L + Rc) + Rp / (1 + s * C * Rp)
    hs = h / 10 # ZOH distorts the input sinusoid by O(ωh); sample finer than the data for this check
    n = 40000
    y = simulate(θpaper4, [0.0, 0.0], sin.(ω .* hs .* (0:n-1)), hs)
    @test (maximum(y[n÷2:end]) - minimum(y[n÷2:end])) / 2 ≈ abs(Z(im * ω)) rtol = 0.01
end

@testset "solvex0 gives a periodic steady state" begin
    _, u1 = stergiopulos_beats(1)
    for (Gm, θ) in models
        sys = Gm(θ, h)
        x0 = solvex0(sys, u1)
        xn = sys.A * lsim(sys, reshape(u1, 1, :); x0=x0).x[:, end] + sys.B * u1[end]
        @test xn ≈ x0 rtol = 1e-10
        @test solvex0(zoh(WindkesselIdentification.ctmodel(Gm, θ), h), u1) ≈ x0
    end
end

@testset "AD gradient matches finite differences" begin
    y, u = stergiopulos_beats(2)
    for (Gm, θ) in models, x0solve in (true, false)
        x00 = solvex0(Gm(θ, h), u)
        id = IdConditions(θ, x00, ones(Int, length(θ)), !x0solve, x0solve, Gm)
        p0 = 1.1 .* WindkesselIdentification.p(θ, x00, id)
        f(p) = cost(p, id, u, y, h)
        fd = map(eachindex(p0)) do i
            δ = 1e-6 * abs(p0[i])
            e = (1:length(p0)) .== i
            (f(p0 + δ * e) - f(p0 - δ * e)) / 2δ
        end
        @test ForwardDiff.gradient(f, p0) ≈ fd rtol = 1e-5
    end
end

@testset "recovers known parameters from noisy simulated data" begin
    _, u = stergiopulos_beats(10)
    rng = Xoshiro(1)
    for (Gm, θ) in models
        y = simulate(θ, solvex0(Gm(θ, h), u), u, h; G_=Gm) .+ 0.5 .* randn(rng, length(u))
        θ0 = 1.3 .* θ
        phat, res = identify(θ0, solvex0(Gm(θ0, h), u), u, y, h;
            θid=ones(Int, length(θ)), x0id=false, x0solve=true, G_=Gm)
        @test phat ≈ θ rtol = 0.02
    end
    # x0 identified as parameters instead of solved; and unconstrained Newton
    θ = θpaper4[1:3]
    x0 = solvex0(G3(θ, h), u)
    y = simulate(θ, x0, u, h; G_=G3) .+ 0.5 .* randn(rng, length(u))
    for θpos in (true, false)
        phat, _ = identify(1.3 .* θ, 0.9 .* x0, u, y, h; θid=[1, 1, 1], x0id=true, G_=G3, θpos=θpos)
        θhat, x0hat = deal(phat, IdConditions(θ, x0, [1, 1, 1], true, false, G3))
        @test θhat ≈ θ rtol = 0.02
        @test x0hat ≈ x0 rtol = 0.02
    end
end

@testset "reproduces Table 2 of the paper" begin
    data = Dict(:human => stergiopulos_beats(20), :invivo => load_pig(:invivo)[1:2], :exvivo => load_pig(:exvivo)[1:2])
    for (source, Gm, θv1, mse, cmp) in table2
        y, u = data[source]
        θ = tomLs(θv1)
        phat, res = identify(θ, solvex0(Gm(θ, h), u), u, y, h;
            θid=ones(Int, length(θ)), x0id=false, x0solve=true, G_=Gm)
        @test res.minimum / length(u) ≈ mse rtol = 0.01 # 3 significant figures, plus the solvex0 fix
        @test phat[cmp] ≈ θ[cmp] rtol = 0.01
    end
end

@testset "reference outputs for ports are current" begin
    include("reference/generate.jl")
    using .Reference
    ref(f) = readdlm(joinpath(@__DIR__, "reference", f), ','; comments=true, header=true)[1]
    @test hcat(reference_simulation()...) ≈ Float64.(ref("simulation.csv")) rtol = 1e-10
    fits = ref("fits.csv")
    for (i, (name, θ, x0, mse)) in enumerate(reference_fits())
        @test fits[i, 1] == name
        @test isequal(isnan.([θ; x0; mse]), isnan.(Float64.(fits[i, 2:end])))
        @test filter(!isnan, [θ; x0; mse]) ≈ filter(!isnan, Float64.(fits[i, 2:end])) rtol = 1e-6
    end
end

@testset "speed" begin
    y, u = stergiopulos_beats(20)
    θ = θpaper4
    md = zoh(wk4(θ), h)
    x0 = solvex0(md, u)
    allocs(f::F, args::Vararg{Any,N}) where {F,N} = (f(args...); @allocated f(args...)) # specialised, so it measures f only
    @test allocs(sse, md, u, y, x0) == 0
    id = IdConditions(θ, x0, [1, 1, 1, 1], false, true)
    # allocations of the cost (parameter unpacking only) must not grow with the data length
    @test allocs(cost, θ, id, u, y, h) == allocs(cost, θ, id, u[1:100], y[1:100], h)
    identify(θ, x0, u, y, h; x0id=false, x0solve=true)
    t = @elapsed identify(θ, x0, u, y, h; x0id=false, x0solve=true)
    @test t < 2 # about 0.1 s on a laptop; this only catches gross regressions
end

@testset "Aqua" begin
    Aqua.test_all(WindkesselIdentification)
end

end
