import numpy as np
from nest_elephant_tvb.Tvb.modify_tvb.Zerlaut import ZerlautAdaptationSecondOrder


#Matteo
# excitatory
excitatory={
        'C_m':200.0,
        't_ref':5.0,
        'V_reset':-65.0,
        'E_L':-65.0,
        'g_L':10.0,
        'I_e':0.0,
        'a':0.0,
        'b':0.0,
        'Delta_T':2.0,
        'tau_w':500.0,
        'V_th':-50.0,
        'E_ex':0.0,
        'tau_syn_ex':5.0,
        'E_in':-80.0,
        'tau_syn_in':5.0,
    'V_peak': 10.0,
    'N_tot':10**4,
    'p_connect':0.05,
    'g':0.2,
    'Q_e':1.0,
    'Q_i':5.0,
}
#inhibitory
inhibitory={
        'C_m':200.0,
        't_ref':5.0,
        'V_reset':-65.0,
        'E_L':-65.,
        'g_L':10.0,
        'I_e':0.0,
        'a':0.0,
        'b':0.0,
        'Delta_T':0.5,
        'tau_w':1.0,
        'V_th':-50.0,
        'E_ex':0.0,
        'tau_syn_ex':5.0,
        'E_in':-80.0,
        'tau_syn_in':5.0,
    'V_peak': 10.0,
    'N_tot':10**4,
    'p_connect':0.05,
    'g':0.2,
    'Q_e':1.0,
    'Q_i':5.0
}
# Matteo function

class Matteo(ZerlautAdaptationSecondOrder) :

    @staticmethod
    def get_fluct_regime_vars(Fe, Fi, Fe_ext, Fi_ext, W, Q_e, tau_e, E_e, Q_i, tau_i, E_i, g_L, C_m, E_L, N_tot, p_connect, g, K_ext_e, K_ext_i):
        """
        Compute the mean characteristic of neurons.
        Inspired from the next repository :
        https://github.com/yzerlaut/notebook_papers/tree/master/modeling_mesoscopic_dynamics
        :param Fe: firing rate of excitatory population
        :param Fi: firing rate of inhibitory population
        :param Fe_ext: external excitatory input
        :param Fi_ext: external inhibitory input
        :param W: level of adaptation
        :param Q_e: excitatory quantal conductance
        :param tau_e: excitatory decay
        :param E_e: excitatory reversal potential
        :param Q_i: inhibitory quantal conductance
        :param tau_i: inhibitory decay
        :param E_i: inhibitory reversal potential
        :param E_L: leakage reversal voltage of neurons
        :param g_L: leak conductance
        :param C_m: membrane capacitance
        :param E_L: leak reversal potential
        :param N_tot: cell number
        :param p_connect: connectivity probability
        :param g: fraction of inhibitory cells
        :return: mean and variance of membrane voltage of neurons and autocorrelation time constant
        """
        # firing rate
        # 1e-6 represent spontaneous release of synaptic neurotransmitter or some intrinsic currents of neurons
        fe = Fe*(1.-g)*p_connect*N_tot + Fe_ext*K_ext_e
        fi = Fi*g*p_connect*N_tot + Fi_ext*K_ext_i

        # conductance fluctuation and effective membrane time constant
        mu_Ge, mu_Gi = Q_e*tau_e*fe, Q_i*tau_i*fi  # Eqns 5 from [MV_2018]
        mu_G = g_L+mu_Ge+mu_Gi  # Eqns 6 from [MV_2018]
        T_m = C_m/mu_G # Eqns 6 from [MV_2018]

        # membrane potential
        mu_V = (mu_Ge*E_e+mu_Gi*E_i+g_L*E_L-W)/mu_G  # Eqns 7 from [MV_2018]
        # post-synaptic membrane potential event s around muV
        U_e, U_i = Q_e/mu_G*(E_e-mu_V), Q_i/mu_G*(E_i-mu_V)
        # Standard deviation of the fluctuations
        # Eqns 8 from [MV_2018]
        sigma_V = np.sqrt(fe*(U_e*tau_e)**2/(2.*(tau_e+T_m))+fi*(U_i*tau_i)**2/(2.*(tau_i+T_m)))
        fe, fi = fe+1e-9, fi+1e-9 # just to insure a non zero division,
        # Autocorrelation-time of the fluctuations Eqns 9 from [MV_2018]
        T_V_numerator = (fe*(U_e*tau_e)**2 + fi*(U_i*tau_i)**2)
        T_V_denominator = (fe*(U_e*tau_e)**2/(tau_e+T_m) + fi*(U_i*tau_i)**2/(tau_i+T_m))
        # T_V = numpy.divide(T_V_numerator, T_V_denominator, out=numpy.ones_like(T_V_numerator),
        #                    where=T_V_denominator != 0.0) # avoid numerical error but not use with numba
        T_V = T_V_numerator/T_V_denominator
        return mu_V, sigma_V+1e-12, T_V


class Matteo_2(Matteo) :
    @staticmethod
    def threshold_func(muV, sigmaV, TvN, P0=0.0, P1=0.0, P2=0.0, P3=0.0, P4=0.0, P5=0.0, P6=0.0, P7=0.0, P8=0.0, P9=0.0,P10=0.0,P11=0.0,P12=0.0,P13=0.0,P14=0.0,P15=0.0,P16=0.0,P17=0.0,P18=0.0,P19=0.0):
        """
        The threshold function of the neurons
        :param muV: mean of membrane voltage
        :param sigmaV: variance of membrane voltage
        :param TvN: autocorrelation time constant
        :param P: Fitted coefficients of the transfer functions
        :return: threshold of neurons
        """
        # Normalization factors page 48 after the equation 4 from [ZD_2018]
        muV0, DmuV0 = -60.0, 10.0
        sV0, DsV0 = 4.0, 6.0
        TvN0, DTvN0 = 0.5, 1.
        V = (muV-muV0)/DmuV0
        S = (sigmaV-sV0)/DsV0
        T = (TvN-TvN0)/DTvN0
        # Eqns 11 from [MV_2018]
        return P0 + P1*V + P2*S + P3*T + P4*V**2 + P5*S**2 + P6*T**2 + P7*V*S + P8*V*T + P9*S*T+\
            P10*V**3 + P11*V**2*S + P12*V**2*T + P13*V*S**2 + P14*V*S*T +P15*V*T**2+\
            P16*S**3 + P17*S**2*T +P18*S*T**2+\
            P19*T**3

    def TF(self, fe, fi, fe_ext, fi_ext, W, P, E_L):
        """
        transfer function for inhibitory population
        Inspired from the next repository :
        https://github.com/yzerlaut/notebook_papers/tree/master/modeling_mesoscopic_dynamics
        :param fe: firing rate of excitatory population
        :param fi: firing rate of inhibitory population
        :param fe_ext: external excitatory input
        :param fi_ext: external inhibitory input
        :param W: level of adaptation
        :param P: Polynome of neurons phenomenological threshold (order 9)
        :param E_L: leak reversal potential
        :return: result of transfer function
        """
        mu_V, sigma_V, T_V = self.get_fluct_regime_vars(fe, fi, fe_ext, fi_ext, W, self.Q_e, self.tau_e, self.E_e,
                                                           self.Q_i, self.tau_i, self.E_i,
                                                           self.g_L, self.C_m, E_L, self.N_tot,
                                                           self.p_connect, self.g,self.K_ext_e,self.K_ext_i)
        # degree 3
        T_V += self.g_L/self.C_m
        V_thre = self.threshold_func(mu_V, sigma_V, T_V*self.g_L/self.C_m,
                                     P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9],
                                    P[10], P[11], P[12], P[13], P[14], P[15], P[16], P[17], P[18], P[19]) # worst
        # V_thre = self.threshold_func(mu_V, sigma_V, T_V*self.g_L/self.C_m,P[0], P[1], P[2], P[3])
        V_thre *= 1e3  # the threshold need to be in mv and not in Volt
        f_out = self.estimate_firing_rate(mu_V, sigma_V, T_V, V_thre)
        return f_out