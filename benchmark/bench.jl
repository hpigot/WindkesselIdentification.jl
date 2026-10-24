# Timings of the cost, its derivatives and a full fit, on the 20-beat Stergiopulos 1999 input.
# Run with: julia --project=benchmark benchmark/bench.jl
using WindkesselIdentification, ControlSystemsBase, ForwardDiff, Chairmarks, Printf

P, Q, h = load_stergiopulos1999()
smooth!(x, n=5) = append!(x, (collect(1:n-1) .* (x[1] - x[end]) / n) .+ x[end])
y, u = repeat(smooth!(copy(P)), 20), repeat(smooth!(copy(Q)), 20)
θ4 = [0.816, 1.238, 0.0571, 0.00571]

# v1.0 approach (c2d + lsim per evaluation), Float64 only: its Dual path no longer runs
function cost_v1(θ, u, y, h)
    sys = c2d(G(θ, h; discretize=false), h, :zoh)
    x0 = solvex0(sys, u)
    ym = vec(lsim(sys, reshape(u, 1, :); x0=x0).y)
    return sum(abs2, y - ym)
end

report(name, b) = @printf("%-34s %10.1f us %10.1f KiB\n", name, 1e6 * b.time, b.bytes / 1024)
println("n = $(length(u)) samples, Julia $(VERSION)")
for (name, Gm, θ) in (("WK2", G2, θ4[1:2]), ("WK3", G3, θ4[1:3]), ("WK4", G, θ4))
    id = IdConditions(θ, solvex0(Gm(θ, h), u), ones(Int, length(θ)), false, true, Gm)
    f(p) = cost(p, id, u, y, h)
    report("$name cost", @b f(θ))
    report("$name gradient", @b ForwardDiff.gradient(f, θ))
    report("$name Hessian", @b ForwardDiff.hessian(f, θ))
    report("$name identify (from 1.3θ)", @b identify(1.3θ, id.x00, u, y, h; θid=id.θid, x0id=false, x0solve=true, G_=Gm) seconds = 3)
end
report("WK4 cost, v1.0 approach (c2d+lsim)", @b cost_v1(θ4, u, y, h))
