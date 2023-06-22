using Pkg
Pkg.add("StructuralIdentifiability")
using StructuralIdentifiability

ode = @ODEmodel(
    c1'(t) = k9 * 0.3 - h9 * c1(t), # XIa = 0.3
    c2'(t) = k10 * c1(t) + k10_ * (k8_9 * c5(t) * c1(t) / (h8_9 + ka * c7(t))) - h10 * c2(t),
    c3'(t) = k2 * c2(t) * c4(t) / (c4(t) + k2m) + k2_ * (k5_10 * c6(t) * c2(t) / (h5_10 + ka * c7(t))) * c4(t) / (c4(t) + k2m_) - h2 * c3(t),
    c4'(t) = -k2 * c2(t) * c4(t) / (c4(t) + k2m) - k2_ * (k5_10 * c6(t) * c2(t) / (h5_10 + ka * c7(t))) * c4(t) / (c4(t) + k2m_),
    c5'(t) = k8 * c3(t) - h8 * c5(t) - ka * c7(t) * (c5(t) + (k8_9 * c5(t) * c1(t) / (h8_9 + ka * c7(t)))),
    c6'(t) = k5 * c3(t) - h5 * c6(t) - ka * c7(t) * (c6(t) + (k5_10 * c6(t) * c2(t) / (h5_10 + ka * c7(t)))),
    c7'(t) = k_apc * c3(t) - h_apc * c7(t),
    c8'(t) = k1 * c3(t),
    y1(t) = c1(t),
    y2(t) = c2(t),
    y3(t) = c3(t),
    y4(t) = c4(t),
    y5(t) = c5(t),
    y6(t) = c6(t),
    y7(t) = c7(t),
    y8(t) = c8(t)
)

# print(assess_identifiability(ode))
print(assess_local_identifiability(ode))
