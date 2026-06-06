using SCA 
using CUDA
using DataFrames, CSV

function test_1(order::Int, ns::Int, nt::Int, ldomain, nl::Int)
    t = CuArray(rand(Float32, nt, ns))
    l = CuArray(UInt8.(rand(UInt8, nt, nl)))
    # m2 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, CuArray, 8}(8, ns, 256)
    # m3 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, CuArray, 16}(8, ns, 256)
    m4 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, CuArray, nl}(order, ns, 256)

    CUDA.@profile Moments.centered_sum_update!(m4, t, l)
    # m4
end

function test_2()
    t = rand(Float32, 100000, 1000)
    l = rand(UInt8, 100000, 16)
    m4 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, Array, 4}(8, 1000, 256)
    @time "CPU moment update" Moments.centered_sum_update!(m4, t, l[:, 1:4])
end

# test_1_wrapper("A5000-atomic-results", [2, 4, 8, 16, 32, 64], [100, 500, 1000], [10000, 20000, 50000, 100000], [2, 4, 8, 16, 32, 64, 128])
# test_1_wrapper("A5000-non-atomic-results", [2, 4, 8, 16, 32, 64], [100, 500, 1000], [10000, 20000, 50000, 100000], [2, 4, 8, 16, 32, 64, 128])
# test_1_wrapper("A5000-non-atomic-results-2", [2, 4, 8, 16, 32], [100, 200, 500, 1000, 2000, 5000], [50000, ], [2, 4, 8, 16, 32, 64, 128])
function test_1_wrapper(output_name::String, orders::Vector{Int}, Ns::Vector{Int}, Nt::Vector{Int}, lsize::Vector{Int})
    # result_file = open("profiling/internal-profiling/$(output_name).csv", "w")
    
    for (order, ns, nt, lsize) in Iterators.product(orders, Ns, Nt, lsize)
        println("Profiling for order=$(order) ns=$(ns) nt=$(nt) lsize=$(lsize)")
        bench_result = test_1(order, ns, nt, 0:255, lsize)
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

# function main()
#     test_1()
# end

# main()