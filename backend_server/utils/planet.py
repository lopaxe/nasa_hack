import numpy as np
import pandas as pd


class Planet:

    def __init__(self, T_star, R, a, M_exo, R_exo):

        self.T_star = T_star
        self.R = R
        self.a = a
        self.A = 0

        self.M_exo = M_exo
        self.R_exo = R_exo

    @staticmethod
    def exp(dT, a, b, c):

        return a * np.exp(-b * dT) + c

    @property
    def T_eq(self):
        R_sol = 6.957e5  # km
        AU = 1.496e8  # km
        K = -272.15  # C

        R = self.R * R_sol
        a = self.a * AU

        return self.T_star * np.sqrt(R/(2*a)) * (1-self.A)**(1/4) + K

    @property
    def G_exo(self):

        M_earth = 5.972e24  # kg
        R_earth = 6.371e6  # m
        g_earth = 9.8
        G = 6.674e-11

        M_exo = self.M_exo * M_earth
        R_exo = self.R_exo * R_earth

        g_exo_abs = G * M_exo / (R_exo ** 2)
        g_exo = g_exo_abs / g_earth

        return g_exo

    @property
    def surviveHeat(self):
        hot = {}
        hot['lim'] = 36.5
        hot['a'] = 6.56979;
        hot['b'] = 0.1425;
        hot['c'] = -1.8737
        hot['extr'] = 100
        cold = {}
        cold['lim'] = -1
        cold['a'] = 6.82646;
        cold['b'] = 0.0844427;
        cold['c'] = 0.318452
        cold['extr'] = -50
        res = {}
        res['T'] = self.T_eq
        if self.T_eq > cold['lim'] and self.T_eq < hot['lim']:
            res['surv'] = True  # survival bool
            res['t_surv'] = None  # survival time, hrs
            res['cod'] = None  # cause of death
            res['wt'] = 0  # image blending weight
        elif self.T_eq < cold['lim']:
            # Model
            dT = abs(self.T_eq - cold['lim'])
            a = cold['a'];
            b = cold['b'];
            c = cold['c']
            t = np.e ** self.exp(dT, a, b, c)
            # Weight
            wrange = abs(cold['extr'] - cold['lim'])
            wt = dT / wrange
            if wt > 1:
                wt = 1
            # Outs
            res['surv'] = False
            res['t_surv'] = t
            res['cod'] = 'cold'
            res['wt'] = wt
        elif self.T_eq > hot['lim']:
            # Model
            dT = abs(self.T_eq - hot['lim'])
            a = hot['a'];
            b = hot['b'];
            c = hot['c']
            t = np.e ** self.exp(dT, a, b, c)
            # Weight
            wrange = abs(hot['extr'] - hot['lim'])
            wt = dT / wrange
            if wt > 1:
                wt = 1
            # Outs
            res['surv'] = False
            res['t_surv'] = t
            res['cod'] = 'hot'
            res['wt'] = wt
        return res

    @property
    def surviveG(self):
        G_extr = 5
        G_micro = 0.8
        G_hyper = 1.2
        res = {}
        res['G'] = self.G_exo
        if self.G_exo < G_micro:
            res['surv'] = True
            res['flag'] = 'micro'
            res['wt'] = 0.5 * self.G_exo / G_micro
        elif self.G_exo < G_hyper:
            res['surv'] = True
            res['flag'] = 'normal'
            res['wt'] = 0.5
        elif self.G_exo < G_extr:
            res['surv'] = True
            res['flag'] = 'hyper'
            res['wt'] = 0.5 + 0.5 * (self.G_exo - G_hyper) / (G_extr - G_hyper)
        else:
            res['surv'] = False
            res['flag'] = 'extreme'
            res['wt'] = 1
        return res

    @property
    def surviveTotal(self):
        G_surv = 0.1  # survival time in hours when in extreme G
        cfg = {}
        cfg['T'] = self.surviveHeat
        cfg['G'] = self.surviveG
        res = {}
        res['T'] = self.T_eq
        res['G'] = self.G_exo
        res['G_flag'] = cfg['G']['flag']
        res['G_wt'] = cfg['G']['wt']
        res['T_wt'] = cfg['T']['wt']
        res['t_surv'] = cfg['T']['t_surv']
        if cfg['T']['surv'] and cfg['G']['surv']:
            res['surv'] = True
            res['cod'] = None
        else:
            res['surv'] = False
            if (not cfg['T']['surv']) and (not cfg['G']['surv']):
                res['t_surv'] = G_surv
                if cfg['T']['cod'] == 'hot':
                    res['cod'] = 'hot&G'
                else:
                    res['cod'] = 'cold&G'
            elif not cfg['G']['surv']:
                res['cod'] = 'G'
                res['t_surv'] = G_surv
            else:
                if cfg['T']['cod'] == 'hot':
                    res['cod'] = 'hot'
                else:
                    res['cod'] = 'cold'
        return res

    @property
    def RGB(self):
        h = 6.626e-34
        c = 2.998e8
        k = 1.38e-23

        def bbr(T, v):
            I = (2 * h * v ** 3 / c ** 2) * 1 / (np.e ** (h * v / (k * T)) - 1)
            return I

        vR = c / 610e-9
        vG = c / 550e-9
        vB = c / 465e-9
        IR = bbr(self.T_star, vR)
        IG = bbr(self.T_star, vG)
        IB = bbr(self.T_star, vB)
        RGB = pd.DataFrame({
            'color': ['R', 'G', 'B'],
            'I': [IR, IG, IB],
        })
        rgb = RGB['I'] / np.max(RGB['I'])
        rgb = list(rgb)
        rgb.reverse()
        return rgb