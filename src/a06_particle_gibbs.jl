using Distributions
using Plots
using StatsBase

function run_particle_gibbs(n_particles, n_steps, process_noise_std, observation_noise_std)
    # 初期状態
    true_state = 0.0

    # 軌道を保存する配列（新規追加）
    trajectories = [Float64[] for _ in 1:n_particles]
    reference_trajectory = nothing  # 初回は参照軌道なし

    # 記録用の配列
    true_states = [true_state]
    observations = [true_state + rand(Normal(0, observation_noise_std))]
    estimated_states = Float64[]

    for iter in 1:3  # Gibbs反復（通常は収束するまで）
        # パーティクルの初期化
        particles = rand(Normal(observations[1], 1.0), n_particles)  # 最初の観測値を中心に初期化
        if reference_trajectory !== nothing
            # 参照軌道がある場合、1つのパーティクルを置き換え
            particles[end] = reference_trajectory[1]
        end

        # 軌道の初期値を記録
        for i in 1:n_particles
            empty!(trajectories[i])
            push!(trajectories[i], particles[i])
        end

        weights = ones(Float64, n_particles)  # Float64型で初期化

        # パーティクルの逐次更新
        for t in 2:n_steps
            # 真の状態の更新（シミュレーション用）
            if iter == 1  # 初回のみ真の状態を更新
                true_state += rand(Normal(0, process_noise_std))
                push!(true_states, true_state)
                push!(observations, true_state + rand(Normal(0, observation_noise_std)))
            end

            # パーティクルの更新（参照軌道以外）
            for i in 1:(reference_trajectory === nothing ? n_particles : n_particles-1)
                particles[i] += rand(Normal(0, process_noise_std))
            end

            # 参照軌道がある場合、その値を維持
            if reference_trajectory !== nothing && t <= length(reference_trajectory)
                particles[end] = reference_trajectory[t]
            end

            # 軌道を記録
            for i in 1:n_particles
                push!(trajectories[i], particles[i])
            end

            # 重みの計算
            for i in 1:n_particles
                weights[i] = pdf(Normal(particles[i], observation_noise_std), observations[t])
            end
            weights ./= sum(weights)  # 正規化

            # 条件付きリサンプリング
            if reference_trajectory !== nothing
                # 参照軌道は必ず選ばれるように
                resampled_indices = vcat(
                    sample(1:(n_particles-1), Weights(weights[1:n_particles-1]), n_particles-1),
                    n_particles
                )
            else
                resampled_indices = sample(1:n_particles, Weights(weights), n_particles)
            end

            particles = particles[resampled_indices]
            trajectories = trajectories[resampled_indices]
        end

        # このイテレーションでの推定値を記録
        if iter == 1
            estimated_states = [mean(getindex.(trajectories, t)) for t in 1:n_steps]
        end

        # 次のイテレーションのための参照軌道を選択
        reference_trajectory = trajectories[sample(1:n_particles, Weights(weights))]
    end

    return true_states, observations, estimated_states
end

# パラメータ設定
n_particles = 100
n_steps = 50
process_noise_std = 0.1
observation_noise_std = 0.5

# パーティクルギブスの実行
true_states, observations, estimated_states = run_particle_gibbs(n_particles, n_steps, process_noise_std, observation_noise_std)

# 結果のプロット
plot(1:n_steps, true_states, label="True State", linewidth=2)
plot!(1:n_steps, observations, label="Observations", seriestype=:scatter, markersize=3)
plot!(1:n_steps, estimated_states, label="Estimated State", linewidth=2)
xlabel!("Time Step")
ylabel!("State")
title!("Particle Gibbs Example")