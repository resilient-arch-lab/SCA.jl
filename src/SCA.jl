module SCA
export SNR, Moments, Utils, Attack, TTest, Hotellings

include("Utils.jl")
using .Utils

include("TestUtils.jl")

include("Moments.jl")
using .Moments

include("SNR.jl")
using .SNR

include("TTest.jl")
using .TTest

include("Attack.jl")
using .Attack

include("Hotellings.jl")
using .Hotellings


end  # module SCA
