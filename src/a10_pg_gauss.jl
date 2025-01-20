using Distributions
using Plots
using Random
using Statistics
using StatsPlots

# データ生成のパラメータ
K = 3      # クラスタ数
N = 100    # データ点数
M = 10     # 粒子数
n_iterations = 1000  # MCMCの反復回数
burn_in = 500       # バーンイン期間

# 真のパラメータ
true_μ = [-2.0, 0.0, 2.0]  # 真の平均
true_σ = [0.5, 0.5, 0.5]   # 真の標準偏差

# 人工データの生成
Random.seed!(123)  # 再現性のため
z_true = rand(1:K, N)  # 真のクラスタ割り当て
x = zeros(N)
for i in 1:N
    x[i] = rand(Normal(true_μ[z_true[i]], true_σ[z_true[i]]))
end

function particle_gibbs(x, K, N, M, n_iterations)
    # パラメータの初期化
    μ = rand(Normal(0, 1), K)
    σ = fill(1.0, K)  # より安定した初期値
    
    # 粒子の初期化
    particles = zeros(Int, N, M)    # N時点 × M個の粒子
    weights = zeros(M)              # 各粒子の重み
    z_samples = zeros(Int, n_iterations, N)
    μ_samples = zeros(n_iterations, K)
    σ_samples = zeros(n_iterations, K)
    
    for iter in 1:n_iterations
        # パラメータ μ のサンプリング
        for k in 1:K
            cluster_points = x[z_samples[max(1,iter-1),:] .== k]
            if length(cluster_points) > 0
                # 事前分布を追加して安定性を向上
                prior_mean = 0.0
                prior_var = 2.0
                post_var = 1.0 / (1.0/prior_var + length(cluster_points)/σ[k]^2)
                post_mean = post_var * (prior_mean/prior_var + sum(cluster_points)/σ[k]^2)
                μ[k] = rand(Normal(post_mean, sqrt(post_var)))
            else
                μ[k] = rand(Normal(0, 2))
            end
        end

        # パラメータ σ のサンプリング
        for k in 1:K
            cluster_points = x[z_samples[max(1,iter-1),:] .== k]
            if length(cluster_points) > 0
                α = 1.0 + length(cluster_points) / 2
                β = 1.0 + sum((cluster_points .- μ[k]).^2) / 2
                σ[k] = sqrt(rand(InverseGamma(α, β)))
            else
                σ[k] = rand(Uniform(0.1, 2.0))
            end
        end

        # 粒子フィルタによる z のサンプリング
        for t in 1:N
            # 先祖のサンプリング（リサンプリング）
            if t > 1
                ancestor_indices = rand(Categorical(weights), M)
                for m in 1:M
                    particles[1:t-1, m] = particles[1:t-1, ancestor_indices[m]]
                end
            end
            
            # 新しい状態のサンプリング
            for m in 1:M
                # 対数尺度で計算
                log_weights = [logpdf(Normal(μ[k], σ[k]), x[t]) for k in 1:K]
                log_weights .-= maximum(log_weights)
                probs = exp.(log_weights)
                probs ./= sum(probs)
                
                if any(isnan.(probs))
                    probs = fill(1.0/K, K)
                end
                
                # クラスタをサンプリング
                particles[t, m] = rand(Categorical(probs))
            end
            
            # 重みの更新
            log_weights = zeros(M)
            for m in 1:M
                log_weights[m] = sum([logpdf(Normal(μ[particles[j,m]], σ[particles[j,m]]), x[j]) 
                                    for j in 1:t])
            end
            log_weights .-= maximum(log_weights)
            weights = exp.(log_weights)
            weights ./= sum(weights)
        end
        
        # 最終的なサンプルを選択
        selected_particle = rand(Categorical(weights))
        z_samples[iter, :] = particles[:, selected_particle]
        μ_samples[iter, :] = μ
        σ_samples[iter, :] = σ
    end
    
    return z_samples, μ_samples, σ_samples
end

# Particle Gibbsを実行
z_samples, μ_samples, σ_samples = particle_gibbs(x, K, N, M, n_iterations)

# 結果の可視化
# 1. データとクラスタ割り当ての可視化
p1 = plot(title="Data and Clustering Results", xlabel="x", ylabel="Density")

# 真のクラスタの分布
for k in 1:K
    cluster_points = x[z_true .== k]
    histogram!(p1, cluster_points, bins=30, alpha=0.3, 
              label="True cluster $k", normalize=:pdf)
end

# 推定クラスタの分布
z_final = round.(Int, vec(mean(z_samples[burn_in+1:end, :], dims=1)))
for k in 1:K
    cluster_points = x[z_final .== k]
    density!(p1, cluster_points, 
            label="Estimated cluster $k", linestyle=:dash)
end

# 2. μ の事後分布の推移
p2 = plot(μ_samples, label=["μ1" "μ2" "μ3"], 
          title="Trace of μ", xlabel="Iteration", ylabel="Value")
hline!(true_μ, label=["True μ1" "True μ2" "True μ3"], linestyle=:dash)

# 3. σ の事後分布の推移
p3 = plot(σ_samples, label=["σ1" "σ2" "σ3"], 
          title="Trace of σ", xlabel="Iteration", ylabel="Value")
hline!(true_σ, label=["True σ1" "True σ2" "True σ3"], linestyle=:dash)

# 全てのプロットを組み合わせて表示
plot(p1, p2, p3, layout=(3,1), size=(800,1000))