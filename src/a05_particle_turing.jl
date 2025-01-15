using Distributions
using Plots
using Turing

# モデルの定義
@model function particle_filter_model(observations, process_noise_std, observation_noise_std)
    n_steps = length(observations)
    
    # 状態変数を配列として定義
    x = Vector{Real}(undef, n_steps)
    
    # 初期状態
    x[1] ~ Normal(0, 1)
    observations[1] ~ Normal(x[1], observation_noise_std)
    
    # 逐次的な状態と観測の生成
    for t in 2:n_steps
        x[t] ~ Normal(x[t-1], process_noise_std)
        observations[t] ~ Normal(x[t], observation_noise_std)
    end
    
    return x
end

# パラメータ設定
n_steps = 50
process_noise_std = 0.1
observation_noise_std = 0.5

# 真の状態と観測の生成
true_states = cumsum(rand(Normal(0, process_noise_std), n_steps))
observations = true_states + rand(Normal(0, observation_noise_std), n_steps)

# モデルのインスタンス化
model = particle_filter_model(observations, process_noise_std, observation_noise_std)

# 推論の実行
chain = sample(model, PG(100), n_steps)

# 結果の抽出と平均の計算
estimated_states = mean([chain[Symbol(:x, "[", i, "]")] for i in 1:n_steps])

# 結果のプロット
plot(1:n_steps, true_states, label="True State", linewidth=2)
plot!(1:n_steps, observations, label="Observations", seriestype=:scatter, markersize=3)
plot!(1:n_steps, estimated_states, label="Estimated State", linewidth=2)
xlabel!("Time Step")
ylabel!("State")
title!("Particle Filter with Turing.jl")