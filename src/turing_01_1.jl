# https://turing.ml/dev/tutorials/00-introduction/

using MCMCChains
using Random
using Turing
using Distributions

Random.seed!(1234)

p_true = 0.5
N = 1000

# data = rand(Bernoulli(p_true), N)

@model function coinflip(; N::Int)
    p ~ Beta(1,1)
    y ~ filldist(Bernoulli(p), N)
end

coinflip(y::AbstractVector{<:Real}) = coinflip(; N=length(y)) | (; y);

model = coinflip(data)

sampler_mcmc = NUTS()

chain = sample(model, sampler_mcmc, 1_000; progress=false)

histogram(
    chain,
    bin=30
)