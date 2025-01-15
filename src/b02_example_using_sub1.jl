# 参照元パッケージは先に存在する必要がある
module sub_b
    export f_sub_b
    function f_sub_b()
        return 1
    end
end