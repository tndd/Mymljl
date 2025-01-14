using Distributions
using FillArrays
using LinearAlgebra
using Random
using StatsPlots
using Turing

# Set a random seed.
Random.seed!(3)

# Define Gaussian mixture model.
w = [0.5, 0.5]
μ = [-3.5, 0.5]

mixturemodel = MixtureModel(
    [MvNormal(Fill(μₖ, 2), I) for μₖ in μ],
    w
)

# We draw the data points.
N = 600
x = rand(mixturemodel, N);

@model function gaussian_mixture_model(x)
    K = 2
    μ ~ MvNormal(Zeros(K), I)
    w ~ Dirichlet(K, 1.0)
    dist_asigned = Categorical(w)

    D, N = size(x)
    dist_clusters = [
        MvNormal(Fill(μₖ, D), I)
        for μₖ in μ
    ]
end