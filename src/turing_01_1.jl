using Distributions
using MCMCChains
using Random
using StatsPlots
using Turing

# https://turing.ml/dev/tutorials/00-introduction/

Random.seed!(1234)

p_true = 0.5
N = 1000

data = rand(Bernoulli(p_true), N)

prior_belief = Beta(1,1)

function updated_belief(prior_belief::Beta, data::AbstractArray{Bool})
    heads = sum(data)
    tails = length(data) - heads
    return Beta(prior_belief.α + heads, prior_belief.β + tails)
end

@gif for n in 1:N
    plot(
        updated_belief(prior_belief, data[1:n]);
        size=(500, 250),
        title="Updated belief after $n observations",
        xlabel="probability of heads",
        ylabel="",
        legend=nothing,
        xlim=(0, 1),
        fill=0,
        α=0.3,
        w=3,
    )
    vline!([p_true])
end

@model function coinflip(; N::Int)
    p ~ Beta(1,1)
    y ~ filldist(Bernoulli(p), N)
end

coinflip(y::AbstractVector{<:Real}) = coinflip(; N=length(y)) | (; y);

model = coinflip(data)

sampler_mcmc = NUTS()

chain = sample(model, sampler_mcmc, 1_000; progress=false)

density(
    chain;
    xlim=(0, 1),
    legend=:best,
    w=2,
    c=:blue
)

# Visualize a green density plot of the posterior distribution in closed-form.
plot!(
    0:0.01:1,
    pdf.(updated_belief(prior_belief, data), 0:0.01:1);
    xlabel="probability of heads",
    ylabel="",
    title="",
    xlim=(0, 1),
    label="Closed-form",
    fill=0,
    α=0.3,
    w=3,
    c=:lightgreen,
)

# Visualize the true probability of heads in red.
vline!([p_true]; label="True probability", c=:red)