using FastQuantiles
using Documenter

makedocs(;
    modules = [FastQuantiles],
    authors = "David MacMahon <davidm@astro.berkeley.edu> and contributors",
    sitename = "FastQuantiles.jl",
    format = Documenter.HTML(;
        canonical = "https://david-macmahon.github.io/FastQuantiles.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(;
    repo = "github.com/david-macmahon/FastQuantiles.jl",
    devbranch = "main",
)
