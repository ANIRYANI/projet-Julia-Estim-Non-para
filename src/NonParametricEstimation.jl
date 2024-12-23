module NonParametricEstimation

# Include submodules
include("kernels.jl")
include("estimation.jl")

# Re-export submodules (optional, for easy access)
using .Kernels
using .Estimation

end