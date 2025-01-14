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
N = 60
x = rand(mixturemodel, N);

scatter(x[1, :], x[2, :]; legend=false, title="Synthetic Dataset")


@model function gaussian_mixture_model(x)
    # Draw the parameters for each of the K=2 clusters from a standard normal distribution.
    K = 2
    μ ~ MvNormal(Zeros(K), I)

    # Draw the weights for the K clusters from a Dirichlet distribution with parameters αₖ = 1.
    w ~ Dirichlet(K, 1.0)
    # Alternatively, one could use a fixed set of weights.
    # w = fill(1/K, K)

    # Construct categorical distribution of assignments.
    distribution_assignments = Categorical(w)

    # (2次元, N)という行列と思いきや逆
    D, N = size(x)
    distribution_clusters = [MvNormal(Fill(μₖ, D), I) for μₖ in μ]

    # Draw assignments for each datum and generate it from the multivariate normal distribution.
    k = Vector{Int}(undef, N)
    for i in 1:N
        k[i] ~ distribution_assignments
        x[:, i] ~ distribution_clusters[k[i]]
    end

    return k
end

model = gaussian_mixture_model(x);

# サンプラーの設定を変更
sampler = Gibbs(
    PG(10, :k),  # 粒子数を減らす
    HMC(0.05, 10, :μ, :w)  # ステップサイズとステップ数を調整
)

# サンプリングの実行
nsamples = 100
nchains = 4
burn = 20
chains = sample(model, sampler, MCMCSerial(), nsamples, nchains, discard_initial=burn)

plot(chains[["μ[1]", "μ[2]"]]; legend=true)
