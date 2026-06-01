```
ENABLE_JITPROFILING=1 rocprofv3 --output-directory ./roc-profiling --output-format pftrace --hip-trace --hsa-trace --kernel-trace -- julia ./profile.jl
```