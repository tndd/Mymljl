using Distributions
using Plots
using Random

# パラメータの設定
K = 3  # クラスタ数
N = 100  # データ点数
n_iterations = 1000  # MCMCの反復回数

# データの生成
Random.seed!(123)
μ_true = [ -5.0, 0.0, 5.0 ]  # 真の平均
σ_true = [ 1.0, 1.0, 1.0 ]   # 真の標準偏差
z_true = rand(1:K, N)        # 真のクラスタ割り当て
x = rand.(Normal.(μ_true[z_true], σ_true[z_true]))  # 観測データ

# Particle Gibbsの実装
function particle_gibbs(x, K, N, n_iterations)
    # パラメータの初期化
    μ = rand(Normal(0, 1), K)
    σ = rand(Uniform(0.5, 2.0), K)
    z = rand(1:K, N)
    z_samples = zeros(Int, n_iterations, N)

    for iter in 1:n_iterations
        # パラメータ μ のサンプリング
        for k in 1:K
            μ[k] = rand(Normal(mean(x[z .== k]), σ[k] / sqrt(sum(z .== k))))
        end

        # パラメータ σ のサンプリング
        for k in 1:K
            σ[k] = rand(InverseGamma(1 + sum(z .== k) / 2, 1 + sum((x[z .== k] .- μ[k]).^2) / 2))
        end

    # 隠れ変数 z のサンプリング
    for i in 1:N
        # 対数尺度で計算して数値的安定性を改善
        log_weights = [logpdf(Normal(μ[k], σ[k]), x[i]) for k in 1:K]
        # 最大値を引いて指数をとることで、オーバーフローを防ぐ
        log_weights .-= maximum(log_weights)
        weights = exp.(log_weights)
        # 正規化
        weights ./= sum(weights)

        # NaNをチェックし、問題があれば均等な確率を割り当てる
        if any(isnan.(weights))
            weights = fill(1.0/K, K)
        end

        z[i] = rand(Categorical(weights))
    end

        # サンプルを保存
        z_samples[iter, :] = z
    end

    return z_samples, μ, σ
end

# Particle Gibbsを実行
z_samples, μ, σ = particle_gibbs(x, K, N, n_iterations)

# バーンイン期間を100回と仮定
burn_in = 100

# バーンイン期間後のサンプルを使用してクラスタ割り当てを可視化
z_mean = vec(mean(z_samples[burn_in+1:end, :], dims=1))
scatter(x, z_mean, label="Cluster Assignment", xlabel="x", ylabel="Cluster")