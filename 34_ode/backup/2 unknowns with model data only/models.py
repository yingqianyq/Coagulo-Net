import tensorflow as tf
import numpy as np


def jvp(y, x, v):
    # For more information, see https://github.com/renmengye/tensorflow-forward-ad/issues/2
    u = tf.ones_like(y)  # unimportant
    g = tf.gradients(y, x, grad_ys=u)
    return tf.gradients(g, u, grad_ys=v)


class PINN(tf.keras.Model):

    def __init__(self, TIM, NDM, scale, name="pinn", eps=1):
        super().__init__()

        self.nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(100, activation=tf.tanh),
                tf.keras.layers.Dense(100, activation=tf.tanh),
                tf.keras.layers.Dense(100, activation=tf.tanh),
                tf.keras.layers.Dense(34)
            ]
        )
        self.nn.build(input_shape=[None, 1])
        # self.h_10_TPplus = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.log_h_10_TPplus = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        # self.h_10_TPminus = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.log_h_10_TPminus = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        # self.h_2 = tf.Variable(0.0, dtype=tf.float32, trainable=True)

        # self.k_8_m = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        # self.K_8M_m = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        # self.k_8t_m = tf.Variable(0.0, dtype=tf.float32, trainable=True)

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.eps = tf.constant(eps, tf.float32)

        self.TIM = tf.constant(TIM, tf.float32)
        self.NDM = tf.constant(NDM, tf.float32)
        self.scale = tf.constant(scale, tf.float32)

        self._name = name
    
    def call(self, t):
        return self.nn.call(t)

    # @tf.function
    # def ode(self, t):
    #     u = self.call(t)
    #     v = tf.ones_like(t)
    #     u_t = jvp(u, t, v)[0]

    #     u0_t, u1_t, u2_t, u3_t, u4_t, u5_t, u6_t, u7_t, u8_t, u9_t, u10_t, u11_t, u12_t, u13_t, u14_t, u15_t, u16_t, u17_t, u18_t, u19_t, \
    #     u20_t, u21_t, u22_t, u23_t, u24_t, u25_t, u26_t, u27_t, u28_t, u29_t, u30_t, u31_t, u32_t, u33_t = tf.split(u_t, 34, axis=-1)
    #     u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18, u19, \
    #     u20, u21, u22, u23, u24, u25, u26, u27, u28, u29, u30, u31, u32, u33 = tf.split(u, 34, axis=-1)

    #     # parameters
    #     k_T7_plus = 3.2 * 1e-3         # nM^{-1}s^{-1}    %% binding of TF & VII 0.0458;%
    #     k_T7_minus = 3.1 * 1e-3        # s^{-1}           %% dissociation of TF:VII 0.0183;%
    #     k_T7a_plus = 0.023             # nM^{-1}s^{-1}    %% binding of TF & VIIa 0.0458;%
    #     k_T7a_minus = 3.1 * 1e-3       # s^{-1}           %% dissociation of TF:VIIa 0.0183;%
    #     k_TF7 = 4.4 * 1e-4             # nM^{-1}s^{-1}    %% auto-activation of VII (H&M,2002)
    #     k_10_7 = 0.013                 # nM^{-1}s^{-1}    %% Xa-activation of VII
    #     k_2_7 = 2.3 * 1e-5             # nM^{-1}s^{-1}    %% IIa-activation of VII
    #     k_7_10 = 69.0/60.0             # s^{-1}           %% TF:VIIa activation of X (103.0/60.0; AM,2008) (Mann et al 1990,Blood J)
    #     k_7_10M = 450.0                # nM               %% TF:VIIa activation of X (240.0; AM,2008) (Mann et al 1990)
    #     k_9 = 15.6/60.0                # s^{-1}           %% TF:VIIa activation of IX (32.4/60.0; AM,2008) (Mann et al 1990,Blood J)

    #     K_9M = 243.0                   # nM               %% TF:VIIa activation of IX (24.0; AM,2008) (Mann et al 1990)
    #     k_2t = 7.5 * 1e-6              # nM^{-1}s^{-1}    %% Xa-activation of II
    #     k_8 = 54.0/60.0                # s^{-1}           %% IIa-activation of VIII(194.4/60.0; AM,2008) (Hill-Eubanks & Lollar 1990)(modified 6/20/2016)
    #     K_8M = 147.0                   # nM               %112000;
    #     k_5 = 0.233                    # s^{-1}           %% IIa-activation of V (%27.0/60.0; AM,2008) (Monkovic & Tracy, 1990)(modified 6/20/2016)
    #     K_5M = 71.7                    # nM               %140.5;
    #     k_f = 59.0                     # s^{-1}           %% IIa-activation of fibrinogen (AM,2008)
    #     K_fM = 3160.0                  # nM
    #     # h_10_TPplus = 4.381            # nM^{-1}s^{-1}    %% binding of Xa with TFPI (modified 6/20/2016)
    #     h_10_TPplus = tf.math.exp(self.log_h_10_TPplus)
    #     h_10_TPminus = 5.293 * 1e-8    # s^{-1}          %% dissociation of Xa:TFPI

    #     h_7_TP = 0.05                  # nM^{-1}s^{-1}   %% Xa:TFPI inactivation of TF:VIIa
    #     h_10_AT = 1.83/60.0 * 1e-4     # %(fminc)  %nM^{-1}s^{-1}   %% ATIII inactivation of Xa % (1.83e-04/60.0,Wb,2003); (1.5e-06, HM 2002); AM,2008 0.347/60.0 (Panteleev 2006)
    #     h_9 = 1.34/60.0 * 1e-5         # nM^{-1}s^{-1}   %% ATIII inactivation of IXa % (1.34e-05/60.0,Wb,2003); (4.9e-07, HM 2002); AM,2008 0.0162/60.0 (Panteleev 2006)
    #     h_2 = 1.79 * 1e-4              # %(fminc)   %nM^{-1}s^{-1}   %% ATIII inactivation of IIa % (2.89e-04/60.0,Wb,2003); (7.1e-06, HM 2002); AM,2008 0.0119 11.56e-03 (Panteleev 2006)(modified 6/20/2016)
    #     h_7_AT = 4.5 * 1e-7            # nM^{-1}s^{-1}   %% ATIII inactivation of TF:VIIa (HM 2002 2.3e-07)(Lawson et al.,1993,4.5e-07 no HS, 5.6e-06 with HS)
    #     k_TEN_plus = 0.01              # nM^{-1}s^{-1}   %% binding of IXa^{m} and VIIIa^{m} (modified 6/20/2016)
    #     k_TEN_minus = 5.0 * 1e-3       # s^{-1}          %% dissociation of IXa{m}:VIIIa{m} 0.01;%
    #     k_10 = 500.0/60.0              # s^{-1}          %% IXa:VIIIa activation of X^{m}(2391.0/60.0 AM 2008) (Mann et al 1990) (20, KnF)
    #     K_10M = 63.0                   # nM              %% IXa:VIIIa activation of X^{m}( %160.0 AM 2008)  (Mann et al 1990) (160, KnF)
    #     k_PRO_plus = 0.4               # nM^{-1}s^{-1}   %% binding of Xa^{m} and Va^{m} 0.1;%

    #     k_PRO_minus = 0.2              # s^{-1}          %% dissociation of Xa:Va  0.01;%
    #     k_2 = 1344.0/60.0              # s^{-1}          %% Xa:Va activation of II^{m}(AM 2008)  (1800.0/60.0; % (Krishnaswamy et al 1990)) Revert to original (16/9/15) (30, KnF)
    #     K_2M = 1060.0                  # nM              %% Xa:Va activation of II^{m}(AM 2008) (1000.0; % (Krishnaswamy et al 1990)) (300, KnF)
    #     k_8t_m = 0.023                 # s^{-1}          %% Xa^{m}-activation of VIII^{m}
    #     K_8tM_m = 20.0                 # nM              %% Xa^{m}-activation of VIII^{m}(KnF,2001)
    #     h_8 = 0.0037                   # s^{-1}          %% spontaneous decay of VIIIa
    #     k_5t_m = 0.046                 # s^{-1}          %% Xa^{m}-activation of V^{m}
    #     K_5tM_m = 10.4                 # nM              %% Xa^{m}-activation of V^{m}(KnF,2001)
    #     h_5 = 0.0028                   # s^{-1}          %% spontaneous decay of Va
    #     k_8_m = 0.9                    # s^{-1}          %% IIa^{m} activation of VIII^{m}(AM,2008)(modified 6/20/2016)
    #     K_8M_m = 147.0                 # nM ??????
    #     k_5_m = 0.233                  # s^{-1}          %% IIa^{m} activation of V^{m}(AM,2008)(modified 6/20/2016)
    #     K_5M_m = 71.7                  # nM
    #     k_10_plus = 0.029              # nM^{-1}s^{-1}   %% platelet-binding of fX/Xa (Krishnaswamy et al. 1998)
    #     k_10_minus = 3.3               # sec^{-1}        %% dissociation of fX/Xa from platelets
    #     k_2_plus = 0.01                # nM^{-1}s^{-1}   %% platelet-binding of fII/IIa
    #     k_2_minus = 5.9                # sec^{-1}        %% dissociation of fII/IIa from platelets
    #     k_9_plus = 0.01                # nM^{-1}s^{-1}   %% platelet-binding of fIX/IXa
    #     k_9_minus = 0.0257             # sec^{-1}        %% dissociation of fIX/IXa from platelets
    #     k_8_plus = 4.3 * 1e-3          # nM^{-1}s^{-1}   %% platelet-binding of fVIII/VIIIa (Raut et al 1999)

    #     k_8_minus = 2.46 * 1e-3        # sec^{-1}        %% dissociation of fVIII/VIIIa from platelets (Raut et al 1999)
    #     k_5_plus = 0.057               # nM^{-1}s^{-1}   %% platelet-binding of fV/Va (Krishnaswamy et al. 1998)
    #     k_5_minus = 0.17               # sec^{-1}        %% dissociation of fV/Va from platelets
    #     kpp = 0.3                      # nM^{-1}s^{-1}   %% platelet-activation of platelet (KnF 2001)
    #     kp2 = 0.37;              # s^{-1}          %% thrombin-activation of platelet 5.4/60.0;%

    #     # equations
    #     eq0 = u0_t - (-k_T7_plus*u0*u1 + k_T7_minus*u2 - k_T7a_plus*u0*u3 + k_T7a_minus*u4) # TF
    #     eq1 = u1_t - (-k_T7_plus*u0*u1 + k_T7_minus*u2 - k_TF7*u4*u1 - k_10_7*u10*u1 - k_2_7*u14*u1) # VII
    #     eq2 = u2_t - (k_T7_plus*u0*u1 - k_T7_minus*u2) # TF:VII
    #     eq3 = u3_t - (-k_T7a_plus*u0*u3 + k_T7a_minus*u4 + k_TF7*u4*u1 + k_10_7*u10*u1 + k_2_7*u14*u1) #VIIa
    #     eq4 = u4_t - (k_T7a_plus*u0*u3 - k_T7a_minus*u4 - h_7_TP*u32*u4 - h_7_AT*u33*u4) # TF:VIIa
    #     eq5 = u5_t - ((-k_9*u4*u5)/(K_9M+u5) - k_9_plus*250*u18*u5 + k_9_minus*u7) # IX
    #     eq6 = u6_t - ((k_9*u4*u5)/(K_9M+u5) - k_9_plus*550*u18*u6 + k_9_minus*u8 - h_9*u33*u6) # IXa        
    #     eq7 = u7_t - (k_9_plus*250*u18*u5 - k_9_minus*u7) # IX(m)
    #     eq8 = u8_t - (-k_TEN_plus*u22*u8 + k_TEN_minus*u23 + k_9_plus*550*u18*u6 - k_9_minus*u8) # IXa(m)
    #     eq9 = u9_t - ((-k_7_10*u4*u9)/(k_7_10M + u9) - k_10_plus*2700*u18*u9 + k_10_minus*u11) # X
    #     eq10 = u10_t - ((k_7_10*u4*u9)/(k_7_10M + u9) - h_10_TPplus*u31*u10 + h_10_TPminus*u32 - h_10_AT*u33*u10 - k_10_plus*2700*u18*u10 + k_10_minus*u12) # Xa
    #     eq11 = u11_t - ((-k_10*u23*u11)/(K_10M + u11) + k_10_plus*2700*u18*u9 - k_10_minus*u11) # X(m)
    #     eq12 = u12_t - ((k_10*u23*u11)/(K_10M + u11) - k_PRO_plus*u27*u12 + k_PRO_minus*u28 + k_10_plus*2700*u18*u10 - k_10_minus*u12) # Xa(m)
    #     eq13 = u13_t - (-k_2t*u10*u13 - k_2_plus*2000*u18*u13 + k_2_minus*u15) # II
    #     eq14 = u14_t - (k_2t*u10*u13 - k_2_plus*2000*u18*u14 + k_2_minus*u16 - h_2*u33*u14) # IIa        
    #     eq15 = u15_t - ((-k_2*u28*u15)/(K_2M + u15) + k_2_plus*2000*u18*u13 - k_2_minus*u15) # II(m)
    #     eq16 = u16_t - ((k_2*u28*u15)/(K_2M + u15) + k_2_plus*2000*u18*u14 - k_2_minus*u16) # IIa(m)        
    #     eq17 = u17_t - (-kpp*u17*u18 - (kp2*u17*u14)/(1 + u14)) # PL
    #     eq18 = u18_t - (kpp*u17*u18 + (kp2*u17*u14)/(1 + u14)) # AP
    #     eq19 = u19_t - ((-k_8*u14*u19)/(K_8M + u19) - k_8_plus*750*u18*u19 + k_8_minus*u21) # VIII
    #     eq20 = u20_t - ((k_8*u14*u19)/(K_8M + u19) - k_8_plus*750*u18*u20 + k_8_minus*u22 - h_8*u20) # VIIIa        
    #     eq21 = u21_t - ((-k_8_m*u16*u21)/(K_8M_m + u21) - (k_8t_m*u12*u21)/(K_8tM_m + u21) + k_8_plus*750*u18*u19 - k_8_minus*u21) # VIII(m)  
    #     eq22 = u22_t - ((k_8_m*u16*u21)/(K_8M_m + u21) + (k_8t_m*u12*u21)/(K_8tM_m + u21) + k_8_plus*750*u18*u20 - k_8_minus*u22 - k_TEN_plus*u22*u8 + k_TEN_minus*u23) # VIIIa(m)
    #     eq23 = u23_t - (k_TEN_plus*u22*u8 - k_TEN_minus*u23) # IXa:VIIIa
    #     eq24 = u24_t - ((-k_5*u14*u24)/(K_5M + u24) - k_5_plus*2700*u18*u24 + k_5_minus*u26) # V
    #     eq25 = u25_t - ((k_5*u14*u24)/(K_5M + u24) - k_5_plus*2700*u18*u25 + k_5_minus*u27 - h_5*u25) # Va
    #     eq26 = u26_t - ((-k_5_m*u16*u26)/(K_5M_m + u26) - (k_5t_m*u12*u26)/(K_5tM_m + u26) + k_5_plus*2700*u18*u24 - k_5_minus*u26) # V(m)        
    #     eq27 = u27_t - ((k_5_m*u16*u26)/(K_5M_m + u26) + (k_5t_m*u12*u26)/(K_5tM_m + u26) - k_PRO_plus*u12*u27 + k_PRO_minus*u28 + k_5_plus*2700*u18*u25 - k_5_minus*u27) # Va(m)        
    #     eq28 = u28_t - (k_PRO_plus*u12*u27 - k_PRO_minus*u28) # Xa(m):Va(m)
    #     eq29 = u29_t - ((-k_f*u14*u29)/(K_fM + u29)) # I
    #     eq30 = u30_t - ((k_f*u14*u29)/(K_fM + u29)) # Ia
    #     eq31 = u31_t - (-h_10_TPplus*u10*u31 + h_10_TPminus*u32) # TFPI
    #     eq32 = u32_t - (h_10_TPplus*u10*u31 - h_10_TPminus*u32 - h_7_TP*u4*u32) # xa:TFPI
    #     eq33 = u33_t - (-u33*(h_10_AT*u10 + h_9*u6 + h_2*u14 + h_7_AT*u4)) # ATIII

    #     return tf.concat([eq0, eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14, eq15, eq16, eq17, eq18, eq19, eq20, \
    #     eq21, eq22, eq23, eq24, eq25, eq26, eq27, eq28, eq29, eq30, eq31, eq32, eq33], axis=-1)

    @tf.function
    def ode(self, t):
        u = self.call(t)
        v = tf.ones_like(t)
        u_t = jvp(u, t, v)[0]

        u0_t, u1_t, u2_t, u3_t, u4_t, u5_t, u6_t, u7_t, u8_t, u9_t, u10_t, u11_t, u12_t, u13_t, u14_t, u15_t, u16_t, u17_t, u18_t, u19_t, \
        u20_t, u21_t, u22_t, u23_t, u24_t, u25_t, u26_t, u27_t, u28_t, u29_t, u30_t, u31_t, u32_t, u33_t = tf.split(u_t, 34, axis=-1)
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18, u19, \
        u20, u21, u22, u23, u24, u25, u26, u27, u28, u29, u30, u31, u32, u33 = tf.split(u, 34, axis=-1)

        # parameters
        k_T7_plus = 3.2 * 1e-3         # nM^{-1}s^{-1}    %% binding of TF & VII 0.0458;%
        k_T7_minus = 3.1 * 1e-3        # s^{-1}           %% dissociation of TF:VII 0.0183;%
        k_T7a_plus = 0.023             # nM^{-1}s^{-1}    %% binding of TF & VIIa 0.0458;%
        k_T7a_minus = 3.1 * 1e-3       # s^{-1}           %% dissociation of TF:VIIa 0.0183;%
        k_TF7 = 4.4 * 1e-4             # nM^{-1}s^{-1}    %% auto-activation of VII (H&M,2002)
        k_10_7 = 0.013                 # nM^{-1}s^{-1}    %% Xa-activation of VII
        k_2_7 = 2.3 * 1e-5             # nM^{-1}s^{-1}    %% IIa-activation of VII
        k_7_10 = 69.0/60.0             # s^{-1}           %% TF:VIIa activation of X (103.0/60.0; AM,2008) (Mann et al 1990,Blood J)
        k_7_10M = 450.0                # nM               %% TF:VIIa activation of X (240.0; AM,2008) (Mann et al 1990)
        k_9 = 15.6/60.0                # s^{-1}           %% TF:VIIa activation of IX (32.4/60.0; AM,2008) (Mann et al 1990,Blood J)

        K_9M = 243.0                   # nM               %% TF:VIIa activation of IX (24.0; AM,2008) (Mann et al 1990)
        k_2t = 7.5 * 1e-6              # nM^{-1}s^{-1}    %% Xa-activation of II
        k_8 = 54.0/60.0                # s^{-1}           %% IIa-activation of VIII(194.4/60.0; AM,2008) (Hill-Eubanks & Lollar 1990)(modified 6/20/2016)
        K_8M = 147.0                   # nM               %112000;
        k_5 = 0.233                    # s^{-1}           %% IIa-activation of V (%27.0/60.0; AM,2008) (Monkovic & Tracy, 1990)(modified 6/20/2016)
        K_5M = 71.7                    # nM               %140.5;
        k_f = 59.0                     # s^{-1}           %% IIa-activation of fibrinogen (AM,2008)
        K_fM = 3160.0                  # nM
        # h_10_TPplus = 4.381            # nM^{-1}s^{-1}    %% binding of Xa with TFPI (modified 6/20/2016)
        h_10_TPplus = tf.math.exp(self.log_h_10_TPplus)
        # h_10_TPminus = 5.293 * 1e-8    # s^{-1}          %% dissociation of Xa:TFPI
        h_10_TPminus = tf.math.exp(self.log_h_10_TPminus)

        h_7_TP = 0.05                  # nM^{-1}s^{-1}   %% Xa:TFPI inactivation of TF:VIIa
        h_10_AT = 1.83/60.0 * 1e-4     # %(fminc)  %nM^{-1}s^{-1}   %% ATIII inactivation of Xa % (1.83e-04/60.0,Wb,2003); (1.5e-06, HM 2002); AM,2008 0.347/60.0 (Panteleev 2006)
        h_9 = 1.34/60.0 * 1e-5         # nM^{-1}s^{-1}   %% ATIII inactivation of IXa % (1.34e-05/60.0,Wb,2003); (4.9e-07, HM 2002); AM,2008 0.0162/60.0 (Panteleev 2006)
        h_2 = 1.79 * 1e-4              # %(fminc)   %nM^{-1}s^{-1}   %% ATIII inactivation of IIa % (2.89e-04/60.0,Wb,2003); (7.1e-06, HM 2002); AM,2008 0.0119 11.56e-03 (Panteleev 2006)(modified 6/20/2016)
        h_7_AT = 4.5 * 1e-7            # nM^{-1}s^{-1}   %% ATIII inactivation of TF:VIIa (HM 2002 2.3e-07)(Lawson et al.,1993,4.5e-07 no HS, 5.6e-06 with HS)
        k_TEN_plus = 0.01              # nM^{-1}s^{-1}   %% binding of IXa^{m} and VIIIa^{m} (modified 6/20/2016)
        k_TEN_minus = 5.0 * 1e-3       # s^{-1}          %% dissociation of IXa{m}:VIIIa{m} 0.01;%
        k_10 = 500.0/60.0              # s^{-1}          %% IXa:VIIIa activation of X^{m}(2391.0/60.0 AM 2008) (Mann et al 1990) (20, KnF)
        K_10M = 63.0                   # nM              %% IXa:VIIIa activation of X^{m}( %160.0 AM 2008)  (Mann et al 1990) (160, KnF)
        k_PRO_plus = 0.4               # nM^{-1}s^{-1}   %% binding of Xa^{m} and Va^{m} 0.1;%

        k_PRO_minus = 0.2              # s^{-1}          %% dissociation of Xa:Va  0.01;%
        k_2 = 1344.0/60.0              # s^{-1}          %% Xa:Va activation of II^{m}(AM 2008)  (1800.0/60.0; % (Krishnaswamy et al 1990)) Revert to original (16/9/15) (30, KnF)
        K_2M = 1060.0                  # nM              %% Xa:Va activation of II^{m}(AM 2008) (1000.0; % (Krishnaswamy et al 1990)) (300, KnF)
        k_8t_m = 0.023                 # s^{-1}          %% Xa^{m}-activation of VIII^{m}
        K_8tM_m = 20.0                 # nM              %% Xa^{m}-activation of VIII^{m}(KnF,2001)
        h_8 = 0.0037                   # s^{-1}          %% spontaneous decay of VIIIa
        k_5t_m = 0.046                 # s^{-1}          %% Xa^{m}-activation of V^{m}
        K_5tM_m = 10.4                 # nM              %% Xa^{m}-activation of V^{m}(KnF,2001)
        h_5 = 0.0028                   # s^{-1}          %% spontaneous decay of Va
        k_8_m = 0.9                    # s^{-1}          %% IIa^{m} activation of VIII^{m}(AM,2008)(modified 6/20/2016)
        K_8M_m = 147.0                 # nM ??????
        k_5_m = 0.233                  # s^{-1}          %% IIa^{m} activation of V^{m}(AM,2008)(modified 6/20/2016)
        K_5M_m = 71.7                  # nM
        k_10_plus = 0.029              # nM^{-1}s^{-1}   %% platelet-binding of fX/Xa (Krishnaswamy et al. 1998)
        k_10_minus = 3.3               # sec^{-1}        %% dissociation of fX/Xa from platelets
        k_2_plus = 0.01                # nM^{-1}s^{-1}   %% platelet-binding of fII/IIa
        k_2_minus = 5.9                # sec^{-1}        %% dissociation of fII/IIa from platelets
        k_9_plus = 0.01                # nM^{-1}s^{-1}   %% platelet-binding of fIX/IXa
        k_9_minus = 0.0257             # sec^{-1}        %% dissociation of fIX/IXa from platelets
        k_8_plus = 4.3 * 1e-3          # nM^{-1}s^{-1}   %% platelet-binding of fVIII/VIIIa (Raut et al 1999)

        k_8_minus = 2.46 * 1e-3        # sec^{-1}        %% dissociation of fVIII/VIIIa from platelets (Raut et al 1999)
        k_5_plus = 0.057               # nM^{-1}s^{-1}   %% platelet-binding of fV/Va (Krishnaswamy et al. 1998)
        k_5_minus = 0.17               # sec^{-1}        %% dissociation of fV/Va from platelets
        kpp = 0.3                      # nM^{-1}s^{-1}   %% platelet-activation of platelet (KnF 2001)
        kp2 = 0.37;              # s^{-1}          %% thrombin-activation of platelet 5.4/60.0;%

        # equations, involving u4, u9, u10, u12, u18, u31, u32, u33
        # step 1: query scaling factors
        # TODO: query scaling factors, s4, s9, s10, s12, s18, s31, s32, s33
        # step 2: compute equations

        s4 = self.scale[4]
        s9 = self.scale[9]
        s10 = self.scale[10]
        s12 = self.scale[12]
        s18 = self.scale[18]
        s31 = self.scale[31]
        s32 = self.scale[32]
        s33 = self.scale[33]

        # eq10 = u10_t - ((k_7_10*u4*s4*u9*s9)/(k_7_10M + u9*s9) - h_10_TPplus*u31*s31*u10*s10 + h_10_TPminus*u32*s32 - h_10_AT*u33*s33*u10*s10 - k_10_plus*2700*u18*s18*u10*s10 + k_10_minus*u12*s12) / s10 # Xa
        eq10 = u10_t - self.TIM * ((k_7_10*u4*self.NDM[4]*s4*u9*self.NDM[9]*s9)/(k_7_10M + u9*self.NDM[9]*s9) - h_10_TPplus*u31*self.NDM[31]*s31*u10*self.NDM[10]*s10 + h_10_TPminus*u32*self.NDM[32]*s32 - h_10_AT*u33*self.NDM[33]*s33*u10*self.NDM[10]*s10 - k_10_plus*2700*u18*self.NDM[18]*s18*u10*self.NDM[10]*s10 + k_10_minus*u12*self.NDM[12]*s12) / self.NDM[10] / s10 # Xa
        # eq31 = u31_t - (-h_10_TPplus*u10*s10*u31*s31 + h_10_TPminus*u32*s32) / s31  # TFPI
        eq31 = u31_t - self.TIM * (-h_10_TPplus*u10*self.NDM[10]*s10*u31*self.NDM[31]*s31 + h_10_TPminus*u32*self.NDM[32]*s32) / self.NDM[31] / s31
        # eq32 = u32_t - (h_10_TPplus*u10*s10*u31*s31 - h_10_TPminus*u32*s32 - h_7_TP*u4*s4*u32*s32) / s32 # xa:TFPI
        eq32 = u32_t - self.TIM * (h_10_TPplus*u10*self.NDM[10]*s10*u31*self.NDM[31]*s31 - h_10_TPminus*u32*self.NDM[32]*s32 - h_7_TP*u4*self.NDM[4]*s4*u32*self.NDM[32]*s32)  / self.NDM[32] / s32 # xa:TFPI

        # eq31 = self.TIM * h_10_TPminus*u[:, 32:33]*self.NDM[32]*s32 / self.NDM[31] / s31
        # eq31 = self.TIM * h_10_TPminus*u[:, 32:33]
        # eq31 = u[:, 32:33]
        # eq10 = u10_t - ((k_7_10*u4*u9)/(k_7_10M + u9) - h_10_TPplus*u31*u10 + h_10_TPminus*u32 - h_10_AT*u33*u10 - k_10_plus*2700*u18*u10 + k_10_minus*u12) # Xa
        # eq31 = u31_t - (-h_10_TPplus*u10*u31 + h_10_TPminus*u32) # TFPI
        # eq32 = u32_t - (h_10_TPplus*u10*u31 - h_10_TPminus*u32 - h_7_TP*u4*u32) # xa:TFPI
        return tf.concat([eq10, eq31, eq32], axis=-1)

    def loss_function(self, t_f, f, t_u, u):
        f_pred = self.ode(t_f)
        u_pred = self.call(t_u)
        loss_f = tf.reduce_mean((f_pred - f)**2)
        loss_u = tf.reduce_mean((u_pred - u)**2)
        return loss_f + loss_u

    def train_op(self, t_f, f, t_u, u, eps):
        with tf.GradientTape() as tape:
            u_pred = self.call(t_u)
            # loss_u = tf.reduce_mean((u_pred - u)**2)
            loss_u = 1.0 * tf.reduce_mean((u_pred[:, 4:5] - u[:, 4:5]) ** 2) + \
                     1.0 * tf.reduce_mean((u_pred[:, 9:10] - u[:, 9:10]) ** 2) + \
                     1.0 * tf.reduce_mean((u_pred[:, 10:11] - u[:, 10:11]) ** 2) + \
                     1.0 * tf.reduce_mean((u_pred[:, 12:13] - u[:, 12:13]) ** 2) + \
                     1.0 * tf.reduce_mean((u_pred[:, 18:19] - u[:, 18:19]) ** 2) + \
                     1.0 * tf.reduce_mean((u_pred[:, 31:32] - u[:, 31:32]) ** 2) + \
                     1.0 * tf.reduce_mean((u_pred[:, 32:33] - u[:, 32:33]) ** 2) + \
                     1.0 * tf.reduce_mean((u_pred[:, 33:34] - u[:, 33:34]) ** 2)
            u4_loss = tf.reduce_mean((u_pred[:, 4:5] - u[:, 4:5]) ** 2)
            u9_loss = tf.reduce_mean((u_pred[:, 9:10] - u[:, 9:10]) ** 2)
            u10_loss = tf.reduce_mean((u_pred[:, 10:11] - u[:, 10:11]) ** 2)
            u12_loss = tf.reduce_mean((u_pred[:, 12:13] - u[:, 12:13]) ** 2)
            u18_loss = tf.reduce_mean((u_pred[:, 18:19] - u[:, 18:19]) ** 2)
            u31_loss = tf.reduce_mean((u_pred[:, 31:32] - u[:, 31:32]) ** 2)
            u32_loss = tf.reduce_mean((u_pred[:, 32:33] - u[:, 32:33]) ** 2)
            u33_loss = tf.reduce_mean((u_pred[:, 33:34] - u[:, 33:34]) ** 2)
            f_pred = self.ode(t_f)
            # eq0, eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14, eq15, eq16, eq17, eq18, eq19, \
            # eq20, eq21, eq22, eq23, eq24, eq25, eq26, eq27, eq28, eq29, eq30, eq31, eq32, eq33 = tf.split(f_pred, 34, axis=-1)
            eq10, eq31, eq32 = tf.split(f_pred, 3, axis=-1)
            # loss_f = tf.reduce_mean((f_pred - f)**2)
            loss_f = 1e-5 * tf.reduce_mean(eq10 ** 2) + \
                     1 * tf.reduce_mean(eq31 ** 2) + \
                     1 * tf.reduce_mean(eq32 ** 2) 
            # loss = loss_u + loss_f
            loss = loss_u + eps * loss_f
            eq10_loss = tf.reduce_mean(eq10 ** 2)
            # eq14_loss = tf.reduce_mean(eq14 ** 2)
            eq31_loss = tf.reduce_mean(eq31 ** 2)
            eq32_loss = tf.reduce_mean(eq32 ** 2)
            # eq33_loss = tf.reduce_mean(eq33 ** 2)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, loss_u, loss_f, eq10_loss, eq31_loss, eq32_loss, u4_loss, u9_loss, u10_loss, u12_loss, u18_loss, u31_loss, u32_loss, u33_loss

    def train(self, t_f, f, t_u, u, eps, niter=10000):
        t_f = tf.constant(t_f, tf.float32)
        f = tf.constant(f, tf.float32)
        t_u = tf.constant(t_u, tf.float32)
        u = tf.constant(u, tf.float32)
        eps = tf.constant(eps, tf.float32)
        train_op = tf.function(self.train_op)
        # loss_op = tf.function(
        #     lambda: tf.reduce_mean((self.call(t_u) - u)**2) + tf.reduce_mean((self.ode(t_f) - f)**2)
        # )

        loss = []
        min_loss = 10000
        for it in range(niter):
            total_loss, data_loss, ode_loss, eq10_loss, eq31_loss, eq32_loss, u4_loss, u9_loss, u10_loss, u12_loss, u18_loss, u31_loss, u32_loss, u33_loss = train_op(t_f, f, t_u, u, eps)
            # loss += [train_op(t_f, f, t_u, u).numpy()]
            loss += [total_loss.numpy()]
            if it % 1000 == 0:
                # current_loss = loss_op().numpy()
                current_loss = loss[-1]
                # print(it, current_loss, data_loss.numpy(), ode_loss.numpy(), \
                #       eq10_loss.numpy(), eq14_loss.numpy(), eq31_loss.numpy(), eq32_loss.numpy(), eq33_loss.numpy())
                print(it, current_loss, data_loss.numpy(), ode_loss.numpy(), \
                      eq10_loss.numpy(), eq31_loss.numpy(), eq32_loss.numpy())
                # print(it, current_loss, data_loss.numpy(), ode_loss.numpy(), \
                #       eq10_loss.numpy(), eq31_loss.numpy(), eq32_loss.numpy(), \
                #       u4_loss.numpy(), u9_loss.numpy(), u10_loss.numpy(), u12_loss.numpy(), \
                #       u18_loss.numpy(), u31_loss.numpy(), u32_loss.numpy(), u33_loss.numpy())
                if current_loss < min_loss:
                    min_loss = current_loss
                    self.save_weights(
                        filepath="./checkpoints/"+self.name,
                        overwrite=True,
                    )
        print("min total loss: ", min_loss)
        return loss

    def restore(self, name=None):
        if name is None:
            name = self.name
        self.load_weights("./checkpoints/"+self.name)


class Meta(tf.keras.Model):

    def __init__(self, name="meta"):
        super().__init__()

        self.nn_1 = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(3)
            ]
        )
        self.nn_2 = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(3)
            ]
        )
        self.nn_1.build(input_shape=[None, 1])
        self.nn_2.build(input_shape=[None, 1])

        self.a = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.b = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.c = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.d = tf.Variable(0.0, dtype=tf.float32, trainable=True)

        self.opt = tf.keras.optimizers.Adam()

        self._name = name
    
    @tf.function
    def ode(self, t, nn):
        u = nn(t)
        v = tf.ones_like(t)
        u_t = jvp(u, t, v)[0]
        u1_t, u2_t, u3_t = tf.split(u_t, 3, axis=-1)
        u1, u2, u3 = tf.split(u, 3, axis=-1)
        eq1 = u1_t - self.a * u1 * u3
        eq2 = u2_t - self.b * u2 * u3
        eq3 = u3_t - self.c * u1**2 - self.d * u2**2
        return tf.concat([eq1, eq2, eq3], axis=-1)

    def loss_function(self, t_f, f, t_u, u, nn):
        f_pred = self.ode(t_f, nn)
        u_pred = nn(t_u)
        loss_f = tf.reduce_mean((f_pred - f)**2)
        loss_u = tf.reduce_mean((u_pred - u)**2)
        return loss_f + loss_u

    def train_op(
        self,
        t1_f, 
        f1, 
        t1_u, 
        u1,
        t2_f,
        f2,
        t2_u,
        u2,
    ):
        with tf.GradientTape() as tape:
            loss = self.loss_function(t1_f, f1, t1_u, u1, self.nn_1) + \
                   self.loss_function(t2_f, f2, t2_u, u2, self.nn_2)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2, niter=10000):
        t1_f = tf.constant(t1_f, tf.float32)
        f1 = tf.constant(f1, tf.float32)
        t1_u = tf.constant(t1_u, tf.float32)
        u1 = tf.constant(u1, tf.float32)

        t2_f = tf.constant(t2_f, tf.float32)
        f2 = tf.constant(f2, tf.float32)
        t2_u = tf.constant(t2_u, tf.float32)
        u2 = tf.constant(u2, tf.float32)

        train_op = tf.function(self.train_op)
        loss_op = tf.function(
            lambda: self.loss_function(t1_f, f1, t1_u, u1, self.nn_1) + self.loss_function(t2_f, f2, t2_u, u2, self.nn_2)
        )

        loss = []
        min_loss = 10000
        for it in range(niter):
            loss += [
                train_op(t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2).numpy()
            ]
            if it % 1000 == 0:
                current_loss = loss_op().numpy()
                print(it, current_loss)
                if current_loss < min_loss:
                    min_loss = current_loss
                    self.save_weights(
                        filepath="./checkpoints/"+self.name,
                        overwrite=True,
                    )
        return loss

    def restore(self):
        self.load_weights("./checkpoints/"+self.name)


class Meta2(tf.keras.Model):

    def __init__(self, name="meta2"):
        super().__init__()

        self.nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(6)
            ]
        )
        self.nn.build(input_shape=[None, 1])

        self.a = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.b = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.c = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.d = tf.Variable(0.0, dtype=tf.float32, trainable=True)

        self.opt = tf.keras.optimizers.Adam()

        self._name = name
    
    @tf.function
    def ode(self, t1, t2):
        # the first head
        u = self.nn(t1)[:, 0:3]
        v = tf.ones_like(t1)
        u_t = jvp(u, t1, v)[0]
        u1_t, u2_t, u3_t = tf.split(u_t, 3, axis=-1)
        u1, u2, u3 = tf.split(u, 3, axis=-1)
        eq1 = u1_t - self.a * u1 * u3
        eq2 = u2_t - self.b * u2 * u3
        eq3 = u3_t - self.c * u1**2 - self.d * u2**2
        f1 = tf.concat([eq1, eq2, eq3], axis=-1)

        # the second head
        u = self.nn(t2)[:, 3:6]
        v = tf.ones_like(t2)
        u_t = jvp(u, t2, v)[0]
        u1_t, u2_t, u3_t = tf.split(u_t, 3, axis=-1)
        u1, u2, u3 = tf.split(u, 3, axis=-1)
        eq1 = u1_t - self.a * u1 * u3
        eq2 = u2_t - self.b * u2 * u3
        eq3 = u3_t - self.c * u1**2 - self.d * u2**2
        f2 = tf.concat([eq1, eq2, eq3], axis=-1)
        return f1, f2

    def loss_function(self, t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2):
        f1_pred, f2_pred = self.ode(t1_f, t2_f)
        u1_pred = self.nn(t1_f)[:, 0:3]
        u2_pred = self.nn(t2_f)[:, 3:6]
        loss_f = tf.reduce_mean((f1_pred - f1)**2) + tf.reduce_mean((f2_pred - f2)**2)
        loss_u = tf.reduce_mean((u1_pred - u1)**2) + tf.reduce_mean((u2_pred - u2)**2)
        return loss_f + loss_u

    def train_op(
        self, t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2,
    ):
        with tf.GradientTape() as tape:
            loss = self.loss_function(t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2, niter=10000):
        t1_f = tf.constant(t1_f, tf.float32)
        f1 = tf.constant(f1, tf.float32)
        t1_u = tf.constant(t1_u, tf.float32)
        u1 = tf.constant(u1, tf.float32)

        t2_f = tf.constant(t2_f, tf.float32)
        f2 = tf.constant(f2, tf.float32)
        t2_u = tf.constant(t2_u, tf.float32)
        u2 = tf.constant(u2, tf.float32)

        train_op = tf.function(self.train_op)
        loss_op = tf.function(
            lambda: self.loss_function(t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2,)
        )

        loss = []
        min_loss = 10000
        for it in range(niter):
            loss += [
                train_op(t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2).numpy()
            ]
            if it % 1000 == 0:
                current_loss = loss_op().numpy()
                print(it, current_loss)
                if current_loss < min_loss:
                    min_loss = current_loss
                    self.save_weights(
                        filepath="./checkpoints/"+self.name,
                        overwrite=True,
                    )
        return loss

    def restore(self):
        self.load_weights("./checkpoints/"+self.name)
