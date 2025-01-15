module MainModule
    # 参照元パッケージは先に存在する必要がある
    module sub_b
        export f_sub_b
        function f_sub_b()
            return 1
        end
    end

    module sub_a
        module sub_a1
            # 絶対パス形式での指定には、頭にMainが必要
            using Main.MainModule.sub_b
            function f_sub_a1()
                return 10 + f_sub_b()
            end
        end

        module sub_a2
            # モジュールをロードしなくても、内部は評価され実行される
            println("sub_a2 is EMPTY")
        end
    end

    function f_main()
        # 同階層モジュールはそのまま参照可能
        return sub_a.sub_a1.f_sub_a1()
    end
end

using .MainModule
MainModule.f_main()