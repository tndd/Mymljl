using Distributions
using Plots
using StatsBase  # Weightsを使用するために追加

function run_particle_filter(n_particles, n_steps, process_noise_std, observation_noise_std)
    # 初期状態とパーティクル
    true_state = 0.0  # 真の状態の初期値
    particles = rand(Normal(true_state, 1.0), n_particles)  # パーティクルの初期化

    # 重みの初期化
    weights = ones(n_particles) / n_particles

    # 状態と観測の記録
    true_states = [true_state]
    observations = [true_state + rand(Normal(0, observation_noise_std))]
    estimated_states = [mean(particles)]

    # パーティクルフィルタの実行
    for t in 2:n_steps
        # 真の状態の更新
        true_state += rand(Normal(0, process_noise_std))
        push!(true_states, true_state)

        # 観測の生成
        observation = true_state + rand(Normal(0, observation_noise_std))
        push!(observations, observation)

        # パーティクルの更新
        for i in 1:n_particles
            particles[i] += rand(Normal(0, process_noise_std))
        end

        # 重みの更新
        for i in 1:n_particles
            weights[i] = pdf(Normal(particles[i], observation_noise_std), observation)
        end
        weights /= sum(weights)  # 重みの正規化

        # リサンプリング
        resampled_indices = sample(1:n_particles, Weights(weights), n_particles)
        particles = particles[resampled_indices]

        # 状態の推定
        push!(estimated_states, mean(particles))
    end

    return true_states, observations, estimated_states
end

# パラメータ設定
n_particles = 100
n_steps = 50
process_noise_std = 0.1
observation_noise_std = 0.5

# パーティクルフィルタの実行
true_states, observations, estimated_states = run_particle_filter(n_particles, n_steps, process_noise_std, observation_noise_std)

# 結果のプロット
plot(1:n_steps, true_states, label="True State", linewidth=2)
plot!(1:n_steps, observations, label="Observations", seriestype=:scatter, markersize=3)
plot!(1:n_steps, estimated_states, label="Estimated State", linewidth=2)
xlabel!("Time Step")
ylabel!("State")
title!("Particle Filter Example")