using LinearAlgebra

# まず構造体を定義
struct LSTM
    # 入力ゲートのパラメータ
    Wi::Matrix{Float64}
    bi::Vector{Float64}

    # 忘却ゲートのパラメータ
    Wf::Matrix{Float64}
    bf::Vector{Float64}

    # 出力ゲートのパラメータ
    Wo::Matrix{Float64}
    bo::Vector{Float64}

    # セル状態更新のパラメータ
    Wc::Matrix{Float64}
    bc::Vector{Float64}
end

# 活性化関数
σ(x) = 1 ./ (1 .+ exp.(-x))
softmax(x) = exp.(x) ./ sum(exp.(x))

# その後、create_lstm関数を定義
function create_lstm(input_size::Int, hidden_size::Int)
    # 入力と隠れ状態の結合サイズ
    concat_size = input_size + hidden_size

    LSTM(
        # 入力ゲート
        randn(hidden_size, concat_size) .* 0.01,
        zeros(hidden_size),

        # 忘却ゲート
        randn(hidden_size, concat_size) .* 0.01,
        ones(hidden_size),  # 忘却ゲートは1で初期化

        # 出力ゲート
        randn(hidden_size, concat_size) .* 0.01,
        zeros(hidden_size),

        # セル状態更新
        randn(hidden_size, concat_size) .* 0.01,
        zeros(hidden_size)
    )
end

function forward(lstm::LSTM, x_t::Vector{Float64}, h_prev::Vector{Float64}, c_prev::Vector{Float64})
    # ゲートの計算
    i_t = σ.(lstm.Wi * [x_t; h_prev] + lstm.bi)  # 入力ゲート
    f_t = σ.(lstm.Wf * [x_t; h_prev] + lstm.bf)  # 忘却ゲート
    o_t = σ.(lstm.Wo * [x_t; h_prev] + lstm.bo)  # 出力ゲート

    # 新しい情報の計算
    c̃_t = tanh.(lstm.Wc * [x_t; h_prev] + lstm.bc)

    # メモリセルの更新
    c_t = f_t .* c_prev + i_t .* c̃_t

    # 隠れ状態の更新
    h_t = o_t .* tanh.(c_t)

    return h_t, c_t
end

# 使用例
function example()
    # 設定
    input_size = 10
    hidden_size = 16

    # LSTMの初期化
    lstm = create_lstm(input_size, hidden_size)

    # 初期状態
    h = zeros(hidden_size)
    c = zeros(hidden_size)

    # サンプル入力
    x = randn(input_size)

    # 1ステップ実行
    h_new, c_new = forward(lstm, x, h, c)

    println("入力サイズ: ", length(x))
    println("隠れ状態サイズ: ", length(h_new))
    println("セル状態サイズ: ", length(c_new))
end

# 実行
example()