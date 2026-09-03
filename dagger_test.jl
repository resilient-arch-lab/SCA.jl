using Distributed
addprocs(2; exeflags="--threads=4")  # workers 2 and 3 will be used, each have 4 threads
using Dagger
@everywhere using SCA
using GraphViz, JSON3

Dagger.enable_logging!()

# create distributed dataset 
workers = [2 3]  # worker mapping must be same dimension of Blocks of arrays
traces = rand(Blocks(5000, 1000), Float64, 10000, 2000; assignment=workers)
labels = rand(Blocks(5000, 8), UInt8, 10000, 16; assignment=workers)

# call routine
moments = Moments.centered_sum_update(traces, labels, 256, 4)

# Fetch logs
logs = Dagger.fetch_logs!()
Dagger.disable_logging!()

# write logs
open("dagger-logs.json", "w") do io
    Dagger.show_logs(io, logs, :chrome_trace)
end

# check results
m1 = Moments.UniVarMomentsAccVecLabel{Float64, UInt8, Array, 16}(4, size(traces, 2), 256)
Moments.centered_sum_update!(m1, collect(traces), collect(labels))

# results don't match right now.
if all(.≈(m1.moments, collect(moments)))
    println("Correct results")
else
    if any(.≈(m1.moments, collect(moments)))
        println("Partially incorrect results")
    else
        println("Entirely incorrect results")
    end
end

