using Distributions
using Random
using Plots

# パラメータの設定
K = 5  # カテゴリ数
T = 100  # 時系列の長さ
N = 100  # 粒子数
n_iterations = 1000  # MCMCの反復回数

# 状態方程式と観測方程式のパラメータ
transition_matrix = rand(K, K)  # 状態遷移行列 (K x K)
emission_matrix = rand(K, K)    # 観測行列 (K x K)

# 正規化
transition_matrix ./= sum(transition_matrix, dims=2)
emission_matrix ./= sum(emission_matrix, dims=2)

# 状態変数と観測変数の生成
Random.seed!(123)
z_true = zeros(Int, T)
x = zeros(Int, T)

# 初期状態
z_true[1] = rand(1:K)
x[1] = rand(Categorical(emission_matrix[z_true[1], :]))

# 時系列データの生成
for t in 2:T
    z_true[t] = rand(Categorical(transition_matrix[z_true[t-1], :]))
    x[t] = rand(Categorical(emission_matrix[z_true[t], :]))
end

# Particle Gibbsの実装
function particle_gibbs(x, transition_matrix, emission_matrix, K, T, N, n_iterations)
    # 状態変数の初期化
    z = zeros(Int, T)
    z_samples = zeros(Int, n_iterations, T)

    for iter in 1:n_iterations
        # 各時刻 t で状態変数をサンプリング
        for t in 1:T
            # Particle Filterを使用して状態変数をサンプリング
            particles = zeros(Int, N)
            weights = zeros(N)

            # 初期化
            if t == 1
                particles = rand(1:K, N)  # 初期分布からサンプリング
            else
                # 前回の状態に基づいて粒子をサンプリング
                for i in 1:N
                    particles[i] = rand(Categorical(transition_matrix[z[t-1], :]))
                end
            end

            # 重みの計算
            for i in 1:N
                weights[i] = emission_matrix[particles[i], x[t]]
            end
            weights /= sum(weights)

            # リサンプリング
            indices = rand(Categorical(weights), N)
            particles = particles[indices]

            # 状態変数のサンプリング
            z[t] = rand(particles)
        end

        # サンプルを保存
        z_samples[iter, :] = z
    end

    return z_samples
end

# Particle Gibbsを実行
z_samples = particle_gibbs(x, transition_matrix, emission_matrix, K, T, N, n_iterations)

# バーンイン期間を100回と仮定
burn_in = 100

# バーンイン期間後のサンプルを使用してヒストグラムを描画
t = 50  # 任意の時刻 t を選択
histogram(z_samples[burn_in+1:end, t], bins=K, label="Posterior of z_t", xlabel="z_t", ylabel="Frequency")
vline!([z_true[t]], label="True z_t", linewidth=2, linecolor=:red)