using DataFrames, CSV
using StatsPlots, ColorSchemes, Colors

# plotly()

function plot_threadripper_results()
    results_df = CSV.read("profiling/internal-profiling/threadripper-results.csv", DataFrame)

    #Plot1:
    #   y = time
    #   x = NTraces * NSamples
    #   log10 scale axes

    @df results_df scatter(
        :NTraces .* :NSamples, :Time, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/128 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, xticks=10.0.^(collect(6:10)), yticks=10.0.^(collect(-3:2)),
        xlims=(1e5, 1e10), ylims=(1e-3, 1e3),
        xlabel="Samples in dataset", ylabel="Time (S)",
        legend=:outertopright,
        title="Moment Estimation Benchmarks: 64-t Threadripper",
        dpi=300, size=(1000, 600)
    )
    savefig("profiling/internal-profiling/threadripper-time-scatter.png")

    results_df = CSV.read("profiling/internal-profiling/threadripper-results-2.csv", DataFrame)
    @df results_df scatter(
        :NTraces .* :NSamples, :Time, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/128 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, xticks=10.0.^(collect(6:10)), yticks=10.0.^(collect(-3:2)),
        xlims=(1e5, 1e10), ylims=(1e-3, 1e3),
        xlabel="Samples in dataset", ylabel="Time (S)",
        legend=:outertopright,
        title="Moment Estimation Benchmarks: 64-t Threadripper",
        dpi=300, size=(1000, 600)
    )
    savefig("profiling/internal-profiling/threadripper-time-scatter-2.png")

end

function plot_A5000_results()
    results_df = CSV.read("profiling/internal-profiling/A5000-atomic-results.csv", DataFrame)
    
    #Plot1:
    #   y = time
    #   x = NTraces * NSamples
    #   log10 scale axes

    @df results_df scatter(
        :NTraces .* :NSamples, :Time, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/128 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, xticks=10.0.^(collect(6:10)), yticks=10.0.^(collect(-3:2)),
        xlims=(1e5, 1e10), ylims=(1e-3, 1e3),
        xlabel="Samples in dataset", ylabel="Time (S)",
        legend=:outertopright,
        title="Moment Estimation Benchmarks: NVIDIA A5000 (atomic implementation)",
        dpi=300, size=(1000, 600)
    )
    savefig("profiling/internal-profiling/A5000-atomic-time-scatter.png")
    @df results_df scatter(
        :NTraces, :NSamples, :Time, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/128 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, zscale=:log10,
        xlabel="# Traces", ylabel = "# Samples / Trace", zlabel="Time (S)",
        xlims=(1e4, 1e5), ylims=(100, 1000), zlims=(1e-3, 1e3),
        legend=:outertopright,
        title="Moment Estimation Benchmarks: NVIDIA A5000 (atomic implementation)",
        dpi=300, size=(800, 600)
    )
    savefig("profiling/internal-profiling/A5000-atomic-time-scatter-3D.png")

    results_df = CSV.read("profiling/internal-profiling/A5000-non-atomic-results.csv", DataFrame)
    @df results_df scatter(
        :NTraces .* :NSamples, :Time, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/128 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, xticks=10.0.^(collect(6:10)), yticks=10.0.^(collect(-3:2)),
        xlims=(1e5, 1e10), ylims=(1e-3, 1e3),
        xlabel="Samples in dataset", ylabel="Time (S)",
        legend=:outertopright,
        title="Moment Estimation Benchmarks: NVIDIA A5000 (non-atomic implementation)",
        dpi=300, size=(1000, 600)
    )
    savefig("profiling/internal-profiling/A5000-non-atomic-time-scatter.png")
    
    results_df = CSV.read("profiling/internal-profiling/A5000-non-atomic-results-2.csv", DataFrame)
    @df results_df scatter(
        :NSamples, :Time, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/32 .* :Order).^(1/4), 0, 1.0 .* (1/256 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10,
        xlims=(90, 2100), ylims=(1e-3, 1e3),
        xlabel="Samples per trace", ylabel="Time (S) [50k traces]",
        legend=:outertopright,
        title="Moment Estimation Benchmarks: NVIDIA A5000 (non-atomic implementation)",
        dpi=300, size=(1000, 600)
    )
    savefig("profiling/internal-profiling/A5000-non-atomic-time-scatter-2.png")
    
    @df results_df scatter(
        :NTraces, :NSamples, :Time, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/128 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, zscale=:log10,
        xlabel="# Traces", ylabel = "# Samples / Trace", zlabel="Time (S)",
        xlims=(1e4, 1e5), ylims=(100, 1000), zlims=(1e-3, 1e3),
        legend=:outertopright,
        title="Moment Estimation Benchmarks: NVIDIA A5000 (non-atomic implementation)",
        dpi=300, size=(800, 600)
    )
    savefig("profiling/internal-profiling/A5000-non-atomic-time-scatter-3D.png")
end

function plot_intersection_speedup()
    threadripper_df = CSV.read("profiling/internal-profiling/threadripper-results-2.csv", DataFrame)
    a5k_df = CSV.read("profiling/internal-profiling/A5000-atomic-results.csv", DataFrame)

    a5k_intersection = innerjoin(a5k_df, threadripper_df[!, [:Order, :NSamples, :NTraces, :NLabels]], on=[:Order, :NSamples, :NTraces, :NLabels])
    threadripper_intersection = innerjoin(threadripper_df, a5k_df[!, [:Order, :NSamples, :NTraces, :NLabels]], on=[:Order, :NSamples, :NTraces, :NLabels])

    a5k_speedup = threadripper_intersection[:, :Time] ./ a5k_intersection[:, :Time]
    a5k_intersection[!, :Speedup] = a5k_speedup
    
    @df a5k_intersection scatter(
        :NTraces .* :NSamples, :Speedup, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/128 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, xticks=10.0.^(collect(6:10)),
        xlims=(1e5, 1e10),
        xlabel="Samples in dataset", ylabel="Speedup",
        legend=:outertopright,
        title="NVIDIA A5000 speedup (atomic implementation)",
        dpi=300, size=(1000, 600)
    )
    savefig("profiling/internal-profiling/A5000-atomic-speedup-scatter.png")

    
    # Threadripper vs non-atomic results
    threadripper_df = CSV.read("profiling/internal-profiling/threadripper-results-2.csv", DataFrame)
    a5k_df = CSV.read("profiling/internal-profiling/A5000-non-atomic-results.csv", DataFrame)

    a5k_intersection = innerjoin(a5k_df, threadripper_df[!, [:Order, :NSamples, :NTraces, :NLabels]], on=[:Order, :NSamples, :NTraces, :NLabels])
    threadripper_intersection = innerjoin(threadripper_df, a5k_df[!, [:Order, :NSamples, :NTraces, :NLabels]], on=[:Order, :NSamples, :NTraces, :NLabels])

    a5k_speedup = threadripper_intersection[:, :Time] ./ a5k_intersection[:, :Time]
    a5k_intersection[!, :Speedup] = a5k_speedup
    
    @df a5k_intersection scatter(
        :NTraces .* :NSamples, :Speedup, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/128 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, xticks=10.0.^(collect(6:10)),
        xlims=(1e5, 1e10),
        xlabel="Samples in dataset", ylabel="Speedup",
        legend=:outertopright,
        title="NVIDIA A5000 speedup (non-atomic implementation)",
        dpi=300, size=(1000, 600)
    )
    savefig("profiling/internal-profiling/A5000-non-atomic-speedup-scatter.png")

    a5k_intersection[a5k_intersection[:, :Speedup] .>= 1, :]
end