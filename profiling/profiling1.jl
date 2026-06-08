using SCA 
using CUDA
using DataFrames, CSV
using BenchmarkTools
using Statistics

function test_1_CUDA(order::Int, ns::Int, nt::Int, ldomain, lsize::Int)
    t = CuArray(rand(Float32, nt, ns))
    l = CuArray(UInt8.(rand(UInt8, nt, lsize)))
    # m2 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, CuArray, 8}(8, ns, 256)
    # m3 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, CuArray, 16}(8, ns, 256)
    m4 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, CuArray, lsize}(order, ns, 256)

    CUDA.@profile Moments.centered_sum_update!(m4, t, l)
    # m4
end

function test_1_CPU(order::Int, ns::Int, nt::Int, ldomain, lsize::Int)
    t = Array(rand(Float32, nt, ns))
    l = Array(UInt8.(rand(UInt8, nt, lsize)))
    # m2 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, Array, 8}(8, ns, 256)
    # m3 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, Array, 16}(8, ns, 256)
    m4 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, Array, lsize}(order, ns, 256)

    @benchmark Moments.centered_sum_update!($m4, $t, $l)
end

# test_1_CUDA_wrapper("A5000-atomic-results", [2, 4, 8, 16, 32, 64], [100, 500, 1000], [10000, 20000, 50000, 100000], [2, 4, 8, 16, 32, 64, 128])
# test_1_CUDA_wrapper("A5000-non-atomic-results", [2, 4, 8, 16, 32, 64], [100, 500, 1000], [10000, 20000, 50000, 100000], [2, 4, 8, 16, 32, 64, 128])
# test_1_CUDA_wrapper("A5000-non-atomic-results-2", [2, 4, 8, 16, 32], [100, 200, 400, 800, 1600], [20000, ], [2, 4, 8, 16, 32, 64, 128, 256])
function test_1_CUDA_wrapper(output_name::String, orders::Vector{Int}, Ns::Vector{Int}, Nt::Vector{Int}, lsize::Vector{Int})
    # result_file = open("profiling/internal-profiling/$(output_name).csv", "w")
    
    for (order, ns, nt, lsize) in Iterators.product(orders, Ns, Nt, lsize)
        println("Profiling for order=$(order) ns=$(ns) nt=$(nt) lsize=$(lsize)")
        bench_result = test_1_CUDA(order, ns, nt, 0:255, lsize)
        res_buf = IOBuffer()
        show(res_buf, bench_result)
        res_string = String(take!(res_buf))

        gpu_time_start_idx = findfirst("GPU was busy for ", res_string)[end] + 1
        gpu_time_end_idx = findnext("s", res_string, gpu_time_start_idx)[end]
        gpu_time = res_string[gpu_time_start_idx:gpu_time_end_idx]
        
        results = DataFrame(
            "device" => "GPU", 
            "order" => order, 
            "# samples" => ns, 
            "# traces" => nt,
            "# labels" => lsize,
            "time" => gpu_time
        )

        CSV.write("profiling/internal-profiling/$(output_name).csv", results, append=true)
    end
end

# test_1_CPU_wrapper("threadripper-results-3", [2, 4, 8, 16, 32], [100, 200, 400, 800, 1600], [20000, ], [2, 4, 8, 16, 32, 64, 128, 256])
function test_1_CPU_wrapper(output_name::String, orders::Vector{Int}, Ns::Vector{Int}, Nt::Vector{Int}, lsize::Vector{Int})
    # result_file = open("profiling/internal-profiling/$(output_name).csv", "w")
    
    for (order, ns, nt, lsize) in Iterators.product(orders, Ns, Nt, lsize)
        println("Profiling for order=$(order) ns=$(ns) nt=$(nt) lsize=$(lsize)")
        bench_result = test_1_CPU(order, ns, nt, 0:255, lsize)

        cpu_time = mean(bench_result.times) * 1e-9  # time in seconds
        
        results = DataFrame(
            "device" => "CPU", 
            "order" => order, 
            "# samples" => ns, 
            "# traces" => nt,
            "# labels" => lsize,
            "time" => cpu_time
        )

        CSV.write("profiling/internal-profiling/$(output_name).csv", results, append=true)
    end
end

# function main()
#     test_1()
# end

# main()