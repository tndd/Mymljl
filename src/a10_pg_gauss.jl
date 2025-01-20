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
Random.seed!(123)
z_true = rand(1:K, N)
x = zeros(N)
for i in 1:N
    x[i] = rand(Normal(true_μ[z_true[i]], true_σ[z_true[i]]))
end

function particle_gibbs(x, K, N, M, n_iterations)
    # 結果保存用の配列
    z_samples = zeros(Int, n_iterations, N)
    μ_samples = zeros(n_iterations, K)
    σ_samples = zeros(n_iterations, K)
    
    # パラメータの初期化
    μ = rand(Normal(0, 1), K)
    σ = ones(K)
    
    for iter in 1:n_iterations
        # パラメータのサンプリング
        for k in 1:K
            cluster_points = x[z_samples[max(1,iter-1),:] .== k]
            if !isempty(cluster_points)
                # μのサンプリング
                n_k = length(cluster_points)
                post_var = 1.0 / (1.0 + n_k/σ[k]^2)
                post_mean = post_var * sum(cluster_points)/σ[k]^2
                μ[k] = rand(Normal(post_mean, sqrt(post_var)))
                
                # σのサンプリング
                α = 1.0 + n_k/2
                β = 1.0 + sum((cluster_points .- μ[k]).^2)/2
                σ[k] = sqrt(rand(InverseGamma(α, β)))
            end
        end

        # 粒子フィルタ
        particles = zeros(Int, N, M)
        weights = ones(M) / M
        
        for t in 1:N
            # リサンプリング
            if t > 1
                indices = rand(Categorical(weights), M)
                particles[1:t-1, :] = particles[1:t-1, indices]
            end
            
            # 新しい状態のサンプリング
            for m in 1:M
                log_probs = [logpdf(Normal(μ[k], σ[k]), x[t]) for k in 1:K]
                probs = exp.(log_probs .- maximum(log_probs))
                probs ./= sum(probs)
                particles[t, m] = rand(Categorical(probs))
            end
            
            # 重みの更新
            for m in 1:M
                weights[m] = prod(pdf.(Normal.(μ[particles[1:t,m]], σ[particles[1:t,m]]), x[1:t]))
            end
            weights ./= sum(weights)
        end
        
        # 最終的なサンプルを保存
        final_particle = rand(Categorical(weights))
        z_samples[iter, :] = particles[:, final_particle]
        μ_samples[iter, :] = μ
        σ_samples[iter, :] = σ
    end
    
    return z_samples, μ_samples, σ_samples
end

# Particle Gibbsを実行
z_samples, μ_samples, σ_samples = particle_gibbs(x, K, N, M, n_iterations)

# 結果の可視化
colors = [:skyblue, :salmon, :lightgreen]

# データとクラスタ割り当ての可視化
p1 = plot(title="Data and Clustering Results", xlabel="x", ylabel="Density")

# 真のクラスタの分布
for k in 1:K
    cluster_points = x[z_true .== k]
    histogram!(p1, cluster_points, bins=30, alpha=0.3, 
               color=colors[k], fillalpha=0.3,
               label="True cluster $k", normalize=:pdf)
end

# 推定結果の整理
μ_final = vec(mean(μ_samples[burn_in+1:end, :], dims=1))
perm = sortperm(μ_final)
z_final = round.(Int, vec(mean(z_samples[burn_in+1:end, :], dims=1)))

# ラベルの付け直し
z_reordered = zeros(Int, N)
for i in 1:N
    z_reordered[i] = findfirst(perm .== z_final[i])
end

# 推定結果の可視化
for k in 1:K
    cluster_points = x[z_reordered .== k]
    density!(p1, cluster_points, 
            color=colors[k], linestyle=:dash,
            label="Estimated cluster $k")
end

# μ の事後分布の推移
p2 = plot(μ_samples, color=permutedims(colors), 
          label=["μ1" "μ2" "μ3"], 
          title="Trace of μ", xlabel="Iteration", ylabel="Value")
hline!(true_μ, color=permutedims(colors), 
       label=["True μ1" "True μ2" "True μ3"], linestyle=:dash)

# σ の事後分布の推移
p3 = plot(σ_samples, color=permutedims(colors), 
          label=["σ1" "σ2" "σ3"], 
          title="Trace of σ", xlabel="Iteration", ylabel="Value")
hline!(true_σ, color=permutedims(colors), 
       label=["True σ1" "True σ2" "True σ3"], linestyle=:dash)

# 全てのプロットを組み合わせて表示
plot(p1, p2, p3, layout=(3,1), size=(800,1000))