using Distributions
using Plots
using Random

# パラメータの設定
σ = 1.0  # 状態方程式のノイズの標準偏差
τ = 0.5  # 観測方程式のノイズの標準偏差
T = 100  # 時系列の長さ
N = 100  # 粒子数
n_iterations = 1000  # MCMCの反復回数

# 状態変数と観測変数の生成
Random.seed!(123)
z_true = rand(Normal(0, σ), T)  # 真の状態変数
x = rand.(Normal.(z_true, τ))   # 観測変数

# Particle Gibbsの実装
function particle_gibbs(x, σ, τ, T, N, n_iterations)
    # 状態変数の初期化
    z = zeros(T)
    z_samples = zeros(n_iterations, T)

    for iter in 1:n_iterations
        # 各時刻 t で状態変数をサンプリング
        for t in 1:T
            # Particle Filterを使用して状態変数をサンプリング
            particles = zeros(N)
            weights = zeros(N)

            # 初期化
            if t == 1
                particles = rand(Normal(0, σ), N)
            else
                particles = rand(Normal.(z[t-1], σ), N)
            end

            # 重みの計算
            weights = pdf.(Normal.(particles, τ), x[t])
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
z_samples = particle_gibbs(x, σ, τ, T, N, n_iterations)

# 結果の可視化
t = 50  # 任意の時刻 t を選択
println(size(z_samples)) # (1000, 100)　=> 反復数回分のzの変化過程100が格納されている
histogram(z_samples[:, t], bins=30, label="Posterior of z_t", xlabel="z_t", ylabel="Frequency")
vline!([z_true[t]], label="True z_t", linewidth=2, linecolor=:red)