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

@model function gaussian_mixture_model_ordered(x)
    # Draw the parameters for each of the K=2 clusters from a standard normal distribution.
    K = 2
    μ ~ Bijectors.ordered(MvNormal(Zeros(K), I))
    # Draw the weights for the K clusters from a Dirichlet distribution with parameters αₖ = 1.
    w ~ Dirichlet(K, 1.0)
    # Alternatively, one could use a fixed set of weights.
    # w = fill(1/K, K)
    # Construct categorical distribution of assignments.
    distribution_assignments = Categorical(w)
    # Construct multivariate normal distributions of each cluster.
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

model = gaussian_mixture_model_ordered(x);

# サンプラーの設定を変更
sampler = Gibbs(
    PG(10, :k),  # 粒子数を減らす
    HMC(0.05, 10, :μ, :w)  # ステップサイズとステップ数を調整
)

# サンプリングの実行
nsamples = 150
nchains = 4
burn = 10
chains = sample(model, sampler, MCMCThreads(), nsamples, nchains, discard_initial = burn);

plot(chains[["μ[1]", "μ[2]"]]; legend=true)
plot(chains[["w[1]", "w[2]"]]; legend=true)

# Model with mean of samples as parameters.
μ_mean = [mean(chains, "μ[$i]") for i in 1:2]
w_mean = [mean(chains, "w[$i]") for i in 1:2]
mixturemodel_mean = MixtureModel([MvNormal(Fill(μₖ, 2), I) for μₖ in μ_mean], w_mean)
contour(
    range(-7.5, 3; length=1_000),
    range(-6.5, 3; length=1_000),
    (x, y) -> logpdf(mixturemodel_mean, [x, y]);
    widen=false,
)
scatter!(x[1, :], x[2, :]; legend=false, title="Synthetic Dataset")