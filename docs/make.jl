using Documenter, WindkesselIdentification

DocMeta.setdocmeta!(WindkesselIdentification, :DocTestSetup, :(using WindkesselIdentification); recursive=true)

makedocs(
    sitename="WindkesselIdentification.jl",
    modules=[WindkesselIdentification],
    authors="Henry Pigot",
    format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://hpigot.github.io/WindkesselIdentification.jl"),
    pages=["Home" => "index.md", "API" => "api.md"],
    checkdocs=:exports,
)

deploydocs(repo="github.com/hpigot/WindkesselIdentification.jl.git", push_preview=false)
