module Kernels

# Noyau Gaussien
function gaussian_kernel(x::Real)
    return exp(-x^2 / 2) / sqrt(2π)
end

# Noyau d'Epanechnikov 
function epanechnikov_kernel(x::Real)
    return abs(x) <= 1 ? 0.75 * (1 - x^2) : 0
end

# Noyau rectangulaire
function uniform_kernel(x::Real)
    return abs(x) <= 1 ? 0.5 : 0
end

# Noyau triangulaire
function triangular_kernel(x::Real)
    return abs(x) <= 1 ? (1 - abs(x)) : 0
end

export gaussian_kernel, epanechnikov_kernel, uniform_kernel, triangular_kernel

end

