module SCA
export SNR, Moments, Utils, Attack


include("SNR.jl")
using .SNR

include("Moments.jl")
using .Moments

include("Utils.jl")
using .Utils

include("Attack.jl")
using .Attack 




end  # module SCA
