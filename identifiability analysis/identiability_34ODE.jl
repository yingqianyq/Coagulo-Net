using Pkg
Pkg.add("StructuralIdentifiability")
using StructuralIdentifiability

ode = @ODEmodel(
    x0'(t) = -k_T7_plus*x0(t)*x1(t) + k_T7_minus*x2(t) - k_T7a_plus*x0(t)*x3(t) + k_T7a_minus*x4(t), # TF
    x1'(t) = -k_T7_plus*x0(t)*x1(t) + k_T7_minus*x2(t) - k_TF7*x4(t)*x1(t) - k_10_7*x10(t)*x1(t) - k_2_7*x14(t)*x1(t), # VII
    x2'(t) = k_T7_plus*x0(t)*x1(t) - k_T7_minus*x2(t), # TF:VII
    x3'(t) = -k_T7a_plus*x0(t)*x3(t) + k_T7a_minus*x4(t) + k_TF7*x4(t)*x1(t) + k_10_7*x10(t)*x1(t) + k_2_7*x14(t)*x1(t), #VIIa
    x4'(t) = k_T7a_plus*x0(t)*x3(t) - k_T7a_minus*x4(t) - h_7_TP*x32(t)*x4(t) - h_7_AT*x33(t)*x4(t), # TF:VIIa
    x5'(t) = (-k_9*x4(t)*x5(t))/(k_9M+x5(t)) - k_9_plus*250*x18(t)*x5(t) + k_9_minus*x7(t), # IX
    x6'(t) = (k_9*x4(t)*x5(t))/(k_9M+x5(t)) - k_9_plus*550*x18(t)*x6(t) + k_9_minus*x8(t) - h_9*x33(t)*x6(t), # IXa        
    x7'(t) = k_9_plus*250*x18(t)*x5(t) - k_9_minus*x7(t), # IX(m)
    x8'(t) = -k_TEN_plus*x22(t)*x8(t) + k_TEN_minus*x23(t) + k_9_plus*550*x18(t)*x6(t) - k_9_minus*x8(t), # IXa(m)
    x9'(t) = (-k_7_10*x4(t)*x9(t))/(k_7_10M + x9(t)) - k_10_plus*2700*x18(t)*x9(t) + k_10_minus*x11(t), # X
    x10'(t) = (k_7_10*x4(t)*x9(t))/(k_7_10M + x9(t)) - h_10_TPplus*x31(t)*x10(t) + h_10_TPminus*x32(t) - h_10_AT*x33(t)*x10(t) - k_10_plus*2700*x18(t)*x10(t) + k_10_minus*x12(t), # Xa
    x11'(t) = (-k_10*x23(t)*x11(t))/(K_10M + x11(t)) + k_10_plus*2700*x18(t)*x9(t) - k_10_minus*x11(t), # X(m)
    x12'(t) = (k_10*x23(t)*x11(t))/(K_10M + x11(t)) - k_PRO_plus*x27(t)*x12(t) + k_PRO_minus*x28(t) + k_10_plus*2700*x18(t)*x10(t) - k_10_minus*x12(t), # Xa(m)
    x13'(t) = -k_2t*x10(t)*x13(t) - k_2_plus*2000*x18(t)*x13(t) + k_2_minus*x15(t), # II
    x14'(t) = k_2t*x10(t)*x13(t) - k_2_plus*2000*x18(t)*x14(t) + k_2_minus*x16(t) - h_2*x33(t)*x14(t), # IIa        
    x15'(t) = (-k_2*x28(t)*x15(t))/(K_2M + x15(t)) + k_2_plus*2000*x18(t)*x13(t) - k_2_minus*x15(t), # II(m)
    x16'(t) = (k_2*x28(t)*x15(t))/(K_2M + x15(t)) + k_2_plus*2000*x18(t)*x14(t) - k_2_minus*x16(t), # IIa(m)        
    x17'(t) = (-kpp*x17(t)*x18(t) - kp2*x17(t)*x14(t))/(1 + x14(t)), # PL
    x18'(t) = (kpp*x17(t)*x18(t) + kp2*x17(t)*x14(t))/(1 + x14(t)), # AP
    x19'(t) = (-k_8*x14(t)*x19(t))/(K_8M + x19(t)) - k_8_plus*750*x18(t)*x19(t) + k_8_minus*x21(t), # VIII
    x20'(t) = (k_8*x14(t)*x19(t))/(K_8M + x19(t)) - k_8_plus*750*x18(t)*x20(t) + k_8_minus*x22(t) - h_8*x20(t), # VIIIa        
    x21'(t) = (-k_8_m*x16(t)*x21(t))/(K_8M_m + x21(t)) - (k_8t_m*x12(t)*x21(t))/(K_8tM_m + x21(t)) + k_8_plus*750*x18(t)*x19(t) - k_8_minus*x21(t), # VIII(m)  
    x22'(t) = (k_8_m*x16(t)*x21(t))/(K_8M_m + x21(t)) + (k_8t_m*x12(t)*x21(t))/(K_8tM_m + x21(t)) + k_8_plus*750*x18(t)*x20(t) - k_8_minus*x22(t) - k_TEN_plus*x22(t)*x8(t) + k_TEN_minus*x23(t), # VIIIa(m)
    x23'(t) = k_TEN_plus*x22(t)*x8(t) - k_TEN_minus*x23(t), # IXa:VIIIa
    x24'(t) = (-k_5*x14(t)*x24(t))/(K_5M + x24(t)) - k_5_plus*2700*x18(t)*x24(t) + k_5_minus*x26(t), # V
    x25'(t) = (k_5*x14(t)*x24(t))/(K_5M + x24(t)) - k_5_plus*2700*x18(t)*x25(t) + k_5_minus*x27(t) - h_5*x25(t), # Va
    x26'(t) = (-k_5_m*x16(t)*x26(t))/(K_5M_m + x26(t)) - (k_5t_m*x12(t)*x26(t))/(K_5tM_m + x26(t)) + k_5_plus*2700*x18(t)*x24(t) - k_5_minus*x26(t), # V(m)        
    x27'(t) = (k_5_m*x16(t)*x26(t))/(K_5M_m + x26(t)) + (k_5t_m*x12(t)*x26(t))/(K_5tM_m + x26(t)) - k_PRO_plus*x12(t)*x27(t) + k_PRO_minus*x28(t) + k_5_plus*2700*x18(t)*x25(t) - k_5_minus*x27(t), # Va(m)        
    x28'(t) = k_PRO_plus*x12(t)*x27(t) - k_PRO_minus*x28(t), # Xa(m):Va(m)
    x29'(t) = (-k_f*x14(t)*x29(t))/(K_fM + x29(t)), # I
    x30'(t) = (k_f*x14(t)*x29(t))/(K_fM + x29(t)), # Ia
    x31'(t) = -h_10_TPplus*x10(t)*x31(t) + h_10_TPminus*x32(t), # TFPI
    x32'(t) = h_10_TPplus*x10(t)*x31(t) - h_10_TPminus*x32(t) - h_7_TP*x4(t)*x32(t), # xa:TFPI
    x33'(t) = -x33(t)*(h_10_AT*x10(t) + h_9*x6(t) + h_2*x14(t) + h_7_AT*x4(t)), # ATIII
    y0(t) = x0(t),
    y1(t) = x1(t),
    y2(t) = x2(t),
    y3(t) = x3(t),
    y4(t) = x4(t),
    y5(t) = x5(t),
    y6(t) = x6(t),
    y7(t) = x7(t),
    y8(t) = x8(t),
    y9(t) = x9(t),
    y10(t) = x10(t),
    y11(t) = x11(t),
    y12(t) = x12(t),
    y13(t) = x13(t),
    y14(t) = x14(t) + x16(t),
    y15(t) = x15(t),
    y17(t) = x17(t),
    y18(t) = x18(t),
    y19(t) = x19(t),
    y20(t) = x20(t),
    y21(t) = x21(t),
    y22(t) = x22(t),
    y23(t) = x23(t),
    y24(t) = x24(t),
    y25(t) = x25(t),
    y26(t) = x26(t),
    y27(t) = x27(t),
    y28(t) = x28(t),
    y29(t) = x29(t),
    y30(t) = x30(t),
    y31(t) = x31(t),
    y32(t) = x32(t),
    y33(t) = x33(t)
)

print(assess_local_identifiability(ode))

