# Persistence of excitation of the three flow signals (Fig. 7 of Pigot et al. 2021): the largest singular values
# of the circular autocorrelation matrix of u. That matrix is circulant, so its singular values are the
# periodogram |DFT(u)|^2 / n, sorted. Run from the repository root:
#   julia --project=examples examples/spectrum.jl
include("idanalysis.jl")
using FFTW

flows = (S1999=load_stergiopulos1999()[2], Sinvivo=load_pig(:invivo; segments=1)[2], Sexvivo=load_pig(:exvivo; segments=1)[2])
S = map(u -> (s = sort(abs2.(fft(u)), rev=true); s ./ s[1]), flows)
display(plot(collect(S), xlims=(1, 20), lw=2, marker=:circle, label=permutedims(collect(string.(keys(S)))), xlabel="k", ylabel="σk/σ1"))

writecsv(joinpath(@__DIR__, "output", "spectrum.csv"),
    "the largest 20 normalised singular values of the circular autocorrelation matrix of the flow\nS1999: Stergiopulos et al. (1999) Fig. 4A; Sinvivo, Sexvivo: first segment of Ao_baseline1",
    (x=1:20, (k => s[1:20] for (k, s) in pairs(S))...))
