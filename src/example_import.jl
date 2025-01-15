module MainModule
    module sub_b
        export f_sub_b
        function f_sub_b()
            return 1
        end
    end
    module sub_a
        module sub_a1
            using Main.MainModule.sub_b
            function f_sub_a1()
                return 10 + f_sub_b()
            end
        end
        module sub_a2
            println("sub_a2 is EMPTY")
        end
    end

    function f_main()
        return MainModule.sub_a.sub_a1.f_sub_a1()
    end
end

using .MainModule
MainModule.f_main()