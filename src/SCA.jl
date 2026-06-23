module SCA
export SNR, Moments, MultiVarMoments, Utils, Attack, TTest


include("SNR.jl")
using .SNR

include("TTest.jl")
using .TTest

include("Moments.jl")
using .Moments

include("MultiVarMoments.jl")
using .MultiVarMoments

include("Utils.jl")
using .Utils

include("Attack.jl")
using .Attack 




end  # module SCA
