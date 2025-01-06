using LinearAlgebra

struct SimpleRNN
    # パラメータ
    Wxh::Matrix{Float64}  # 入力層の重み
    Whh::Matrix{Float64}  # 隠れ層の重み
    Why::Matrix{Float64}  # 出力層の重み
    bh::Vector{Float64}   # 隠れ層のバイアス
    by::Vector{Float64}   # 出力層のバイアス
end

# コンストラクタ
function SimpleRNN(input_size::Int, hidden_size::Int, output_size::Int)
    SimpleRNN(
        randn(hidden_size, input_size) .* 0.01,  # Wxh
        randn(hidden_size, hidden_size) .* 0.01, # Whh
        randn(output_size, hidden_size) .* 0.01, # Why
        zeros(hidden_size),                      # bh
        zeros(output_size)                       # by
    )
end

# 活性化関数
tanh_(x) = tanh.(x)

# フォワードパス
function forward(rnn::SimpleRNN, x::Vector{Float64}, h::Vector{Float64})
    # 隠れ層の状態を更新
    h_new = tanh_(rnn.Wxh * x + rnn.Whh * h + rnn.bh)

    # 出力層の計算
    y = softmax(rnn.Why * h_new + rnn.by)

    return y, h_new
end

# シーケンス全体の処理
function process_sequence(rnn::SimpleRNN, inputs::Vector{Vector{Float64}})
    h = zeros(size(rnn.Whh, 1))  # 初期の隠れ状態
    outputs = []

    for x in inputs
        y, h = forward(rnn, x, h)
        push!(outputs, y)
    end

    return outputs
end


# 使用例
input_size = 2
hidden_size = 3
output_size = 2

rnn = SimpleRNN(input_size, hidden_size, output_size)
x = randn(input_size)  # サンプル入力
h = zeros(hidden_size) # 初期隠れ状態
y, h_new = forward(rnn, x, h)