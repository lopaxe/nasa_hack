import pandas as pd

import numpy as np
import openpyxl
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
import cv2
import codecs
import os, shutil


# from transforms import RGBTransform

def exp(dT, a, b, c):
    return a * np.exp(-b * dT) + c


## Calculate physical parameters

def getT(T_star, R, a, A):
    """
    From https://en.wikipedia.org/wiki/Planetary_equilibrium_temperature

    T_eq = T_star * sqrt(R/2a) * (1 - A)^1/4

    Where:
        T_eq - Planetary equilibrium temperature
        T_star - Star's (surface?) temperature
        A - Planet's bond albedo
        R - Radius of the star
        a - orbital distance

    Recommended to use default albedo around 0-10%
    """
    R_sol = 6.957e5  # km
    AU = 1.496e8  # km
    K = -272.15  # C
    R *= R_sol
    a *= AU
    T_eq = T_star * np.sqrt(R / (2 * a)) * (1 - A) ** (1 / 4)
    T_eq += K
    return (T_eq)


def getG(M_exo, R_exo):
    """
    Input mass; radius
    Output g_exo at surface

    From https://en.wikipedia.org/wiki/Surface_gravity
    g = G * M_exo / R_exo^2
    """
    M_earth = 5.972e24  # kg
    R_earth = 6.371e6  # m
    g_earth = 9.8
    G = 6.674e-11
    M_exo *= M_earth
    R_exo *= R_earth
    g_exo_abs = G * M_exo / (R_exo ** 2)
    g_exo = g_exo_abs / g_earth  # relative
    return (g_exo)


## Calculate survival & image params
def calibrateHeat():
    """
    TC = (TF - 32) / 1.8
    Survive heat if TC <= 36.5
    Survive cold if TC > -1
    """
    lim_hot = 36.5
    lim_cold = -1

    calib_cold = pd.read_excel("death_calibration.xlsx",
                               sheet_name='Cold')
    calib_hot = pd.read_excel("death_calibration.xlsx",
                              sheet_name='Hot')
    calib_cold['T_C'] = (calib_cold['T_F'] - 32) / 1.8
    calib_hot['T_C'] = (calib_hot['T_F'] - 32) / 1.8

    fit_hot = calib_hot[['T_C', 't_h']].loc[calib_hot['T_C'] > lim_hot]
    fit_cold = calib_cold[['T_C', 't_h']].loc[calib_cold['T_C'] < lim_cold]
    fit_hot['dT'] = fit_hot['T_C'] - lim_hot
    fit_cold['dT'] = - (fit_cold['T_C'] - lim_cold)

    # Fit
    phot, pcov = curve_fit(exp, fit_hot['dT'], np.log(fit_hot['t_h']))
    pcold, pcov = curve_fit(exp, fit_cold['dT'], np.log(fit_cold['t_h']))

    # Plot
    """
    plt.scatter(fit_cold['dT'],np.log(fit_cold['t_h']),label='cold')
    plt.scatter(fit_hot['dT'],np.log(fit_hot['t_h']),label='hot')
    plt.plot(fit_hot['dT'], exp(fit_hot['dT'], *phot),label='hot-fit')
    plt.plot(fit_cold['dT'], exp(fit_cold['dT'], *pcold),label='cold-fit')
    plt.legend()
    """
    return (phot, pcold)


def surviveHeat(T_eq):
    """
    Input temperature
    Dead or alive; heat flag; survival time; weight coefficient 0-1

    Survival function is exponential on dT:
    a * np.exp(-b * x) + c

    """
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
    res['T'] = T_eq
    if T_eq > cold['lim'] and T_eq < hot['lim']:
        res['surv'] = True  # survival bool
        res['t_surv'] = None  # survival time, hrs
        res['cod'] = None  # cause of death
        res['wt'] = 0  # image blending weight
    elif T_eq <= cold['lim']:
        # Model
        dT = abs(T_eq - cold['lim'])
        a = cold['a'];
        b = cold['b'];
        c = cold['c']
        t = np.e ** exp(dT, a, b, c)
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
    elif T_eq >= hot['lim']:
        # Model
        dT = abs(T_eq - hot['lim'])
        a = hot['a'];
        b = hot['b'];
        c = hot['c']
        t = np.e ** exp(dT, a, b, c)
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
    return (res)


def surviveG(G_exo):
    """
    Input G
    Output alive or dead; gravity flag; weight coefficient 0-1
    """
    G_extr = 5
    G_micro = 0.8
    G_hyper = 1.2
    res = {}
    res['G'] = G_exo
    if G_exo < G_micro:
        res['surv'] = True
        res['flag'] = 'micro'
        res['wt'] = 0.5 * G_exo / G_micro
    elif G_exo < G_hyper:
        res['surv'] = True
        res['flag'] = 'normal'
        res['wt'] = 0.5
    elif G_exo < G_extr:
        res['surv'] = True
        res['flag'] = 'hyper'
        res['wt'] = 0.5 + 0.5 * (G_exo - G_hyper) / (G_extr - G_hyper)
    else:
        res['surv'] = False
        res['flag'] = 'extreme'
        res['wt'] = 1
    return (res)


def surviveTotal(T, G):
    """
    Input gravity/heat dicts
    Output full result dict
    """
    G_surv = 0.1  # survival time in hours when in extreme G
    cfg = {}
    cfg['T'] = surviveHeat(T)
    cfg['G'] = surviveG(G)
    res = {}
    res['T'] = T
    res['G'] = G
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
    return (res)


def getRGB(T):
    """
    Input star temperature
    RGB reflecting black body radiation

    From: https://en.wikipedia.org/wiki/Black-body_radiation
    I(v,T) = 2hv^3/c^2 * 1/(e^(hv/kT) - 1)
    """
    h = 6.626e-34
    c = 2.998e8
    k = 1.38e-23

    def bbr(T, v):
        I = (2 * h * v ** 3 / c ** 2) * 1 / (np.e ** (h * v / (k * T)) - 1)
        return (I)

    vR = c / 610e-9
    vG = c / 550e-9
    vB = c / 465e-9
    IR = bbr(T, vR)
    IG = bbr(T, vG)
    IB = bbr(T, vB)
    RGB = pd.DataFrame({
        'color': ['R', 'G', 'B'],
        'I': [IR, IG, IB],
    })
    rgb = RGB['I'] / np.max(RGB['I'])
    return (rgb)


## Exoplanet database
def loadData():
    df = pd.read_excel("utils/NASA_Dataset_final.xlsx",
                  sheet_name='Final Dataset',
                  header=3)
    #df = df[df.keys()[:9]] # cleans rows a bit
    return(df)


def getSimilar(df, T, G, A):
    """
    Calculate similarity score of planets in database;
    Temperature, G, Albedo;
    Appends similarity score array;
    Finds index of highest similarity
    """
    K = -272.15
    T -= K
    df['pl_teq'] = np.nan
    df['pl_g'] = np.nan
    df['sim'] = np.nan
    for i in df.index:
        # T
        T_exo = getT(df['st_teff'][i],
                     df['st_rad'][i],
                     df['pl_orbsmax'][i],
                     A)
        df.loc[i, 'pl_teq'] = T_exo
        T_exo -= K
        T_sim = 1 - abs((T - T_exo) / np.max((T, T_exo)))
        # G
        G_exo = getG(df['pl_masse'][i], df['pl_rade'][i])
        df.loc[i, 'pl_g'] = G_exo
        G_sim = 1 - abs((G - G_exo) / np.max((G, G_exo)))
        # Total
        df.loc[i, 'sim'] = np.sqrt(T_sim * G_sim)
    idx = df.loc[df['sim'] == np.max(df['sim'])].index[0]
    return (df, idx)


## Tint
def tint(T, img_path):
    rgb = getRGB(T)
    rgb = list(rgb)
    img = cv2.imread(img_path)
    img = np.einsum('ijk,k->ijk', img, rgb)
    img = np.asarray(img, dtype=int)
    return (img)


## Text
def returnText(cfg, T_star):
    # Survival
    text = f"""
    Here's a picture of you on the planet that you've built yourself! 
    It orbits a star that 
    """
    if T_star < 2000:
        text += "appears red "
    elif T_star < 5000:
        text += "has an orange tint "
    elif T_star < 9000:
        text += "has a yellow tint "
    elif T_star < 10300:
        text += "appears white to the human eye "
    else:
        text += "has a bluish color "
    text += f"because of its temperature. "
    text += f"""
    The average temperature on this planet is {round(cfg['T']):.0f} 
    degrees centigrade which is considered 
    """
    if cfg['cod']:
        if 'hot' in cfg['cod']:
            if cfg['T_wt'] == 1:
                text += "extremely hot. "
            else:
                text += "hot. "
        elif 'cold' in cfg['cod']:
            if cfg['T_wt'] == 1:
                text += "extremely cold. "
            else:
                text += "cold. "
    else:
        text += "normal. "
    text += "The surface gravity of your planet is "
    if cfg['G_flag'] == 'normal':
        text += f"around that of the Earth which is normal to the human body. "
    elif cfg['G_flag'] == 'micro':
        if cfg['G'] < 0.1:
            text += f"less than 10% "
        else:
            text += f"around {round(cfg['G'], 1) * 100:.0f}% "
        text += "of that on the Earth and is considered small. "
    elif cfg['G_flag'] == 'hyper':
        if round(cfg['G']) == 1:
            G = 2
        else:
            G = round(cfg['G'])
        text += f"around {G:.0f} times larger than Earth's and is considered strong. "
    elif cfg['G_flag'] == 'extreme':
        G = round(cfg['G'])
        text += f"around {G:.0f} times larger than Earth's and is considered deadly. "
    if cfg['surv']:
        text += f"""
        Congratulations, you would probably survive on this planet!
        """
    else:
        text += "Unfortunately you "
        if cfg['t_surv'] < 0.2:
            text += "wouldn't last 10 minutes "
        elif round(cfg['t_surv']) <= 1:
            text += "would be dead after an hour "
        else:
            text += "would survive for only around "
            if cfg['t_surv'] <= 10:
                text += f"{cfg['t_surv']:.0f} hours "
            elif cfg['t_surv'] <= 50:
                text += f"{round(cfg['t_surv'] / 10, 1) * 10:.0f} hours"
            else:
                text += f"{round(cfg['t_surv'] / 10):.0f} days "
        text += "on this planet "
        if round(cfg['t_surv']) <= 1:
            text += f"because of "
        else:
            text += f"and would eventually die from "
        if 'hot' in cfg['cod']:
            if cfg['T_wt'] == 1:
                text += "extreme heat"
            else:
                text += "too much heat"
        elif 'cold' in cfg['cod']:
            if cfg['T_wt'] == 1:
                text += "extreme cold"
            else:
                text += "hypothermia"
        if '&' in cfg['cod']:
            text += " and "
        if 'G' in cfg['cod']:
            text += "unbearable gravitational force"
        text += "."
    text = text.replace('\n', '').replace("    ", '')
    paragraph1 = text

    # Exoplanet
    df = loadData()
    df, idx = getSimilar(df, cfg['T'], cfg['G'], 0.05)
    planet = df.loc[idx]
    text = f"There is a known exoplanet called '{planet['pl_name']}' "
    text += f"that is similar to your planet with a score of "
    text += f"{planet['sim'] * 100:.0f}. "
    text += f"This planet completes an orbit around its host star "
    text += f"{planet['st_name']} "
    if round(planet['pl_orbper']) == 1:
        text += 'each day.'
    elif planet['pl_orbper'] > 1:
        text += f"every {round(planet['pl_orbper']):.0f} days."
    elif round(planet['pl_orbper'] * 24) == 1:
        text += "every hour."
    elif planet['pl_orbper'] * 24 < 1:
        text += f"multiple times every hour."
    else:
        text += f"every {round(planet['pl_orbper'] * 24):.0f} hours."
    text = text.replace('\n', '').replace("    ", '')
    text = paragraph1 + '\n\n' + text
    return (text)


def hack_index(text_replacement):
    with codecs.open("templates/backup/index.html") as file:
        text = file.read()

    replaced_text = text.replace("""<p class="card-text"> Your survival score: </p>
            <p class="card-text"> Survival status: </p>
            <p class="card-text"> Temperature on your planet: </p>
            <p class="card-text"> Surface gravity on your plannet: </p>
            <p class="card-text"> Similar exoplanet discovered by NASA: </p>""",
                                 f"""<p class="card-text"> {text_replacement} </p>""")

    with open("templates/backup/temp/index.html", "w") as file:
        file.write(replaced_text)

    shutil.copyfile("templates/backup/temp/index.html", "templates/index.html")
