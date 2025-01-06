using LinearAlgebra

struct DeepRNN
    # 第1層
    Wₓₕ₁::Matrix{Float64}  # 入力 → 第1隠れ層
    Wₕ₁ₕ₁::Matrix{Float64} # 第1隠れ層 → 第1隠れ層

    # 第2層
    Wₕ₁ₕ₂::Matrix{Float64} # 第1隠れ層 → 第2隠れ層
    Wₕ₂ₕ₂::Matrix{Float64} # 第2隠れ層 → 第2隠れ層

    # 出力層
    Wₕ₂ᵧ::Matrix{Float64}  # 第2隠れ層 → 出力

    # バイアス項
    bₕ₁::Vector{Float64}   # 第1隠れ層のバイアス
    bₕ₂::Vector{Float64}   # 第2隠れ層のバイアス
    bᵧ::Vector{Float64}    # 出力層のバイアス
end

# コンストラクタ
function DeepRNN(input_size::Int, hidden_size₁::Int, hidden_size₂::Int, output_size::Int)
    DeepRNN(
        randn(hidden_size₁, input_size) .* 0.01,   # Wₓₕ₁
        randn(hidden_size₁, hidden_size₁) .* 0.01, # Wₕ₁ₕ₁
        randn(hidden_size₂, hidden_size₁) .* 0.01, # Wₕ₁ₕ₂
        randn(hidden_size₂, hidden_size₂) .* 0.01, # Wₕ₂ₕ₂
        randn(output_size, hidden_size₂) .* 0.01,  # Wₕ₂ᵧ
        zeros(hidden_size₁),                       # bₕ₁
        zeros(hidden_size₂),                       # bₕ₂
        zeros(output_size)                         # bᵧ
    )
end

# 活性化関数
softmax(x) = exp.(x) ./ sum(exp.(x))

# フォワードパス
function forward(rnn::DeepRNN, x::Vector{Float64}, h₁::Vector{Float64}, h₂::Vector{Float64})
    # 第1層の計算
    h₁_new = tanh.(rnn.Wₓₕ₁ * x + rnn.Wₕ₁ₕ₁ * h₁ + rnn.bₕ₁)

    # 第2層の計算
    h₂_new = tanh.(rnn.Wₕ₁ₕ₂ * h₁_new + rnn.Wₕ₂ₕ₂ * h₂ + rnn.bₕ₂)

    # 出力層の計算
    y = softmax(rnn.Wₕ₂ᵧ * h₂_new + rnn.bᵧ)

    return y, h₁_new, h₂_new
end

# シーケンス全体の処理
function process_sequence(rnn::DeepRNN, inputs::Vector{Vector{Float64}})
    h₁ = zeros(size(rnn.Wₕ₁ₕ₁, 1))  # 第1層の初期隠れ状態
    h₂ = zeros(size(rnn.Wₕ₂ₕ₂, 1))  # 第2層の初期隠れ状態
    outputs = []

    for x in inputs
        y, h₁, h₂ = forward(rnn, x, h₁, h₂)
        push!(outputs, y)
    end

    return outputs
end

# 使用例
input_size = 10
hidden_size₁ = 20
hidden_size₂ = 15
output_size = 5

rnn = DeepRNN(input_size, hidden_size₁, hidden_size₂, output_size)
x = randn(input_size)     # サンプル入力
h₁ = zeros(hidden_size₁)  # 第1層の初期隠れ状態
h₂ = zeros(hidden_size₂)  # 第2層の初期隠れ状態
y, h₁_new, h₂_new = forward(rnn, x, h₁, h₂)