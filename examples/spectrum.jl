using DataFrames, CSV, FFTW, LinearAlgebra, ToeplitzMatrices, Plots

us = [] # flows
push!(us, CSV.read("stergiopulos1999_data/aflow.csv", DataFrame, header=false,types=[Float64, Float64]).Column2)
push!(us, CSV.read("invivo_data/Ao_baseline1_AoQ1.csv", DataFrame, comment="#",header=true,types=[Time, Float64], normalizenames=true).vb_va)
push!(us, CSV.read("exvivo_data/Ao_baseline1_AoQ1.csv", DataFrame, comment="#",header=true,types=[Time, Float64], normalizenames=true).vb_va)

S = [] # normalized singular values
for u in us
    N = length(u)
    D = fft(Matrix{Float64}(I,N,N),1)
    r = 1/N*conj(D)*(abs.(D*u).^2)
    Φ = Array(Toeplitz(r,r))
    Snorm = svd(Φ).S./maximum(svd(Φ).S) 
    display(plot(Snorm, xlims=(1,20), lw=2, marker=:circle, grid=true))
    push!(S,Snorm)
end

# export data for pgfplots
comment = *("# The largest 20 singular values (normalized) of the autocorrelation matrix of flow data from the following sources\n",
"# S1999: parsed at 200Hz (n=166) from stergiopulos1999total doi = {10.1152/ajpheart.1999.276.1.H81}, Figure 4A type A beat using https://automeris.io/WebPlotDigitizer\n",
"# Sinvivo: measured in vivo Ao_baseline1 20201201\n",
"# Sexvivo: measured ex vivo Ao_baseline1 20201202\n")
filename = "spectrum.csv"
open(filename, "w") do f
    write(f, comment)
end
CSV.write(filename, DataFrame(x = collect(1:20), S1999 = S[1][1:20], Sinvivo = S[2][1:20], Sexvivo = S[3][1:20]), header=true, append=true)
# CSV.write("flow1999.csv", DataFrame(u1999 = u[1]))
# CSV.write("flowinvivo.csv", DataFrame(uinvivo = u[2]))
# CSV.write("flowexvivo.csv", DataFrame(uexvivo = u[3]))