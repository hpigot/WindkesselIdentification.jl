# Reference outputs for ports of this package (e.g. a Python one) to test against.
# Regenerate with `julia --project=. test/reference/generate.jl`; runtests.jl checks they still hold.
module Reference
using WindkesselIdentification, DelimitedFiles
export reference_simulation, reference_fits

const θref = [0.79, 1.22, 0.056, 0.0051] # Stergiopulos et al. (1999), Table 1, type A beat
const θstart = [0.85, 1.5, 0.05, 0.006]  # start of the fits below
const models = ((G2, "wk2", 2), (G3, "wk3", 3), (G, "wk4", 4)) # constructor, name, number of parameters

"Pressure from each model with θref on the Stergiopulos beat, from the periodic initial state."
function reference_simulation()
    _, q, h = load_stergiopulos1999()
    sims = map(models) do (Gm, _, n)
        θ = θref[1:n]
        simulate(θ, solvex0(Gm(θ, h), q), q, h; G_=Gm)
    end
    return h .* (0:length(q)-1), q, sims...
end

"Fits of each model to the Stergiopulos beat from θstart, with x0 solved from periodicity."
function reference_fits()
    p, q, h = load_stergiopulos1999()
    map(models) do (Gm, name, n)
        θ0 = θstart[1:n]
        θ, res = identify(θ0, solvex0(Gm(θ0, h), q), q, p, h; θid=ones(Int, n), x0id=false, x0solve=true, G_=Gm)
        x0 = solvex0(Gm(θ, h), q)
        (name, [θ; fill(NaN, 4 - n)], [x0; fill(NaN, 2 - length(x0))], res.minimum / length(q))
    end
end

const dir = @__DIR__
const header = """
# WindkesselIdentification.jl reference outputs. Units: mmHg, mL/s, s; θ = [Rp, C, Rc, L] in
# mmHg s/mL, mL/mmHg, mmHg s/mL, mmHg s^2/mL. Data: load_stergiopulos1999() (one beat, h = 0.005 s).
# x0 is the periodic initial state from solvex0: x1 the volume on C [mL], x2 = L i_L (WK4 only).
"""

if abspath(PROGRAM_FILE) == @__FILE__
    open(joinpath(dir, "simulation.csv"), "w") do io
        print(io, header)
        println(io, "# Pressure with θ = $θref (WK3: first 3, WK2: first 2), x0 = solvex0.")
        println(io, "t,q,p_wk2,p_wk3,p_wk4")
        writedlm(io, hcat(reference_simulation()...), ',')
    end
    open(joinpath(dir, "fits.csv"), "w") do io
        print(io, header)
        println(io, "# identify(...; x0id=false, x0solve=true) from θ0 = $θstart (truncated per model).")
        println(io, "# MSE = sum of squared pressure errors / number of samples [mmHg^2]. NaN: not in the model.")
        println(io, "model,Rp,C,Rc,L,x0_1,x0_2,MSE")
        for (name, θ, x0, mse) in reference_fits()
            println(io, join([name; θ; x0; mse], ','))
        end
    end
end

end
