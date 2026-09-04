using Distributed
worker_procs = addprocs(2; exeflags="--threads=8")  # workers 2 and 3 will be used, each have 4 threads
using Dagger
@everywhere using SCA
using GraphViz, JSON3

Dagger.enable_logging!()

# create distributed dataset
workers = [2 3]  # worker mapping must be same dimension of Blocks of arrays
traces = rand(Blocks(5000, 1000), Float64, 10000, 1000; assignment=workers)
labels = rand(Blocks(5000, 8), UInt8, 10000, 16; assignment=workers)

# call routine
scope = Dagger.scope(workers=worker_procs)
moments = Dagger.with_options(scope=scope) do
    Moments.centered_sum_update(traces, labels, 256, 4)
end

# check results
m1 = Moments.UniVarMomentsAccVecLabel{Float64, UInt8, Array, 16}(4, size(traces, 2), 256)
Moments.centered_sum_update!(m1, collect(traces), collect(labels))

correct = .≈(m1.moments, collect(moments))
if all(correct)
    println("Correct results")
else
    if any(correct)
        println("Partially incorrect results")
        println("Indices of incorrect results:")
        display(findall(correct .== 0))
        println("Error of incorrect results:")
        display(collect(moments)[correct.==0] .- m1.moments[correct.==0])
    else
        println("Entirely incorrect results")
    end
end

# Fetch logs
logs = Dagger.fetch_logs!()
Dagger.disable_logging!()

# write logs
open("dagger-logs.json", "w") do io
    Dagger.show_logs(io, logs, :chrome_trace)
end