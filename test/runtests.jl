using WindkesselIdentification, Test, ControlSystemsBase, ForwardDiff, LinearAlgebra, Random, Dates, DelimitedFiles, Aqua
using WindkesselIdentification: wk2, wk3, wk4, zoh, sse

const h = 0.005
# Paper fits (Pigot et al. 2021, via controlled-cardiac-afterload θwk4/θwk2), in v1.0 units:
# R [mmHg/(L/min)], C [(L/min) s/mmHg], L [mmHg s/(L/min)]; converted here to mL/s.
const θpaper4 = [13.6 * 60 / 1000, 0.0743 * 1000 / 60, 0.952 * 60 / 1000, 0.0952 * 60 / 1000]
const θpaper2 = [13.6 * 60 / 1000, 0.0996 * 1000 / 60]

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
end

@testset "reproduces the paper fit on Stergiopulos 1999" begin
    y, u = stergiopulos_beats(20)
    for (Gm, θ) in ((G, θpaper4), (G2, θpaper2))
        phat, res = identify(θ, solvex0(Gm(θ, h), u), u, y, h;
            θid=ones(Int, length(θ)), x0id=false, x0solve=true, G_=Gm)
        @test phat ≈ θ rtol = 0.005 # paper reports 3 significant figures
    end
end

@testset "speed" begin
    y, u = stergiopulos_beats(20)
    θ = θpaper4
    md = zoh(wk4(θ), h)
    x0 = solvex0(md, u)
    sse(md, u, y, x0)
    @test (@allocated sse(md, u, y, x0)) == 0
    id = IdConditions(θ, x0, [1, 1, 1, 1], false, true)
    us, ys = u[1:100], y[1:100]
    cost(θ, id, u, y, h)
    # allocations of the cost (parameter unpacking only) must not grow with the data length
    @test (@allocated cost(θ, id, u, y, h)) == (@allocated cost(θ, id, us, ys, h))
    identify(θ, x0, u, y, h; x0id=false, x0solve=true)
    t = @elapsed identify(θ, x0, u, y, h; x0id=false, x0solve=true)
    @test t < 2 # about 0.1 s on a laptop; this only catches gross regressions
end

@testset "Aqua" begin
    Aqua.test_all(WindkesselIdentification)
end

end
