module SCA
export SNR, Moments, Utils, Attack, TTest, Hotellings


include("SNR.jl")
using .SNR

include("TTest.jl")
using .TTest

include("Moments.jl")
using .Moments

include("Utils.jl")
using .Utils

include("Attack.jl")
using .Attack 

include("TestUtils.jl")

include("Hotellings.jl")
using .Hotellings


end  # module SCA
