module MainModule
    include("b02_example_using_sub1.jl")
    include("b02_example_using_sub2.jl")

    function f_main()
        # 同階層モジュールはそのまま参照可能
        return sub_a.sub_a1.f_sub_a1()
    end
end

using .MainModule
MainModule.f_main()