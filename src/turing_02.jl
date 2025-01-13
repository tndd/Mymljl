using Distributions
using FillArrays
using LinearAlgebra
using Random
using StatsPlots

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

scatter(x[1, :], x[2, :]; legend=false, title="Synthetic Dataset")