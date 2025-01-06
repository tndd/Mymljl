using LinearAlgebra

struct SimpleNN
    # 層間の重み
    Wₓₕ::Matrix{Float64}  # 入力 → 隠れ層
    Wₕᵧ::Matrix{Float64}  # 隠れ層 → 出力

    # バイアス項
    bₕ::Vector{Float64}   # 隠れ層のバイアス
    bᵧ::Vector{Float64}   # 出力層のバイアス
end

# コンストラクタ
function SimpleNN(input_size::Int, hidden_size::Int, output_size::Int)
    SimpleNN(
        randn(hidden_size, input_size) .* 0.01,  # Wₓₕ
        randn(output_size, hidden_size) .* 0.01, # Wₕᵧ
        zeros(hidden_size),                      # bₕ
        zeros(output_size)                       # bᵧ
    )
end

# 活性化関数
softmax(x) = exp.(x) ./ sum(exp.(x))

# フォワードパス
function forward(nn::SimpleNN, x::Vector{Float64})
    # 隠れ層の計算
    h = tanh.(nn.Wₓₕ * x + nn.bₕ)

    # 出力層の計算
    y = softmax(nn.Wₕᵧ * h + nn.bᵧ)

    return y
end

# 使用例
input_size = 10
hidden_size = 20
output_size = 5

nn = SimpleNN(input_size, hidden_size, output_size)
x = randn(input_size)  # サンプル入力
y = forward(nn, x)