
import math
import numpy as np
from scipy import integrate


PLANCK_C = 6.6261e-34
BOLTZMANN = 1.3806e-23
SPEED_LIGHT = 299792458
SIGMA = 5.6704e-08


def planck(f, t):
    """
    Planck function for blackbody radiation

    Parameters:
        f    Frequencies [Hz]
        t    A temperature [K]
    
    Returns:
        Planck function values [W/(m2*Hz*sr)]
    """

    a = 2 * PLANCK_C / SPEED_LIGHT**2
    b1 = PLANCK_C / BOLTZMANN

    b = a * f**3 / ( np.exp( b1*f/t ) - 1 )
    return b

def vmr2nd(vmr, p, t):
    """
    Derives molecular number densities from volume mixing ratios

    Parameters:
        vmr  Volume mixing ratios [-]
        p    A pressure [Pa]
        t    A temperature [K]
    
    Returns:
        Number densities [1/m3]
    """
    kb = 1.3806e-23

    n = (vmr*p/(kb*t))
    return n

def absorption_coeff(vmr, p, t, xsec):
    """
    Calculates absorption coefficients for one altitude
    
    Parameters:
        vmr  Volume mixing ratios [-]
        p    A pressure [Pa]
        t    A temperature [K]
        xsec Absorption cross-sections [m2]. Dimensions (vmr,f)
    
    Returns:
        Absorption coefficients [1/m]. Dimension (f)

    """
    n = vmr2nd(vmr, p, t)
    # Add an axis to n to make multiplication work
    a = np.multiply(n[:, np.newaxis], xsec)
    a = sum(a, 0)
    return a

def spectral_radiance(f, z, p, t, vmr, xsec, za):
    """
    Calculates the spectral radiance at the top of the atmosphere

    Parameters:
        f     Frequencies [Hz]
        z     Altitudes [m]
        p     Pressures [Pa]
        t     Temperatures [K]
        vmr   Volume mixing ratios [-]. Dimensions (gas, altitudes)
        xsec  Absorption cross-sections [m2]. Dimensions (gas,vmr,f)
        za    Zenith angle [rad].
    
    Returns:
        Spectral radiance [W/(m2*Hz*sr)]. Dimension (f)
    """
    # Init rs to surface blackbody radiation
    rs = planck(f, t[0])
    tau_out = np.zeros(xsec.shape[1:])

    # Loop altitudes
    a_this = 0 # Dummy value
    #
    a_out = np.zeros(xsec.shape[1:])
    for i, z_i in enumerate(z):
        # Absorption at previous level
        a_old = a_this
        
        # Calculate absorption for this level
        a_this = absorption_coeff(vmr[:, i], p[i], t[i], xsec[:, :, i])
        a_out[:,i] = a_this
        
        # We only do radiance transfer from i=1
        if i > 0:
            # Optical thickness of layer
            tau = (a_old + a_this)/2 * ( (z_i - z[i-1]) / np.cos(za) )
            tau_out[:,i] = tau

            # Transmission of layer
            transmission = np.exp(-tau)
            
            # Effective Planck function of the layer
            b = planck(f, (t[i-1]+t[i])/2)
            
            # Update I
            rs = rs*transmission + b*(1 - transmission)
    return rs

data = np.load("C:\\Users\\Arik\\Desktop\\olr_data.npz")

#%%

f = data["f"]
#frekvens, vektor
wn = data["wn"]
#vågtal, vektor
z = data["z"]
#höjd, vektor
p = data["p"]
#atmosfäriskt tryck, vektor
t = data["t"]
#atmosfärisk temperatur, vektor
vmr = data["vmr"]
#volymandelar, matris
xsec = data["xsec"]
#absorptionstvärsnitt, matris


#%% 

pi=np.pi
x=np.linspace(0, np.pi)
y=np.sin(x)
print(np.trapz(y, x))

def ekv_2(v, t, vmr):
    #azi = np.linspace(0, 2* np.pi, num= 3500, endpoint= True)
    #funktion = np.trapz(azi) #integralen av fi
    theta = np.linspace(0, np.pi/2, num = 3500, endpoint = True)
    
    lista=[]
    for i in theta:
        I = spectral_radiance(v, z, p, t, vmr, xsec, i)
        g = I*math.cos(i) * math.sin(i)
        lista.append(g)
    
        
    
    lista = np.array(lista)
    
    a = np.trapz(lista, theta, axis=0)
    b = 2*pi*a
    return b

svar = ekv_2(f, t, vmr)

import matplotlib.pyplot as plt
plt.plot(wn, svar)

#%% 

def ekv_3 (svar, v):
    k = np.trapz(svar, v, axis=0)
    return k

svar2 = ekv_3(svar, f)
print(svar2)

#%% 

za = np.linspace(0, np.pi/2, num = 3500, endpoint = True)
plt.plot(wn, spectral_radiance(f, z, p, t, vmr, xsec, za[1]), color = 'pink', label = 'låg zenitvinkel')
plt.plot(wn, spectral_radiance(f, z, p, t, vmr, xsec, za[3498]), color = 'hotpink', label = 'hög zenitvinkel')
plt.title('Spektral radians som funktion av vågtal')
plt.ylabel('Spektral radians [W/(m2 Hz sr)]')
plt.xlabel('Vågtal [cm-1]')
plt.legend()
#plt.savefig('radians.png', dpi = 300)
plt.show()

#%% 

plt.plot(wn, ekv_2(f, t, vmr), color = 'palevioletred')
plt.title('Spektral emittans som funktion av vågtal')
plt.ylabel('Spektral emittans [W/(m^2 Hz]')
plt.xlabel('Vågtal [cm-1]')
#plt.savefig('emittans.png', dpi = 300)


#%% 

E = np.trapz(svar, f)
print(E)

#%%

min_t = t.argmin()

def perturb_t(t, dt):
    ny_t = []
    index = 0
    for q in t:
        if index <= min_t:
            t1 = q + dt
            ny_t.append(t1)
        else: 
            ny_t.append(q)
        index += 1
    return np.array(ny_t)

t_ny = perturb_t(t, 1)



def perturb_vmr(vmr, igas, dvmr, t):
    vmr_ny = vmr.copy()
    x = np.array(igas)
    element = np.array(range(0, 33))
    vmr_ny[x, element] = vmr[x, element] * (1 + dvmr)
    return vmr_ny

#%%

diff_t = ekv_2(f, t_ny, vmr) - ekv_2(f, t, vmr)
plt.plot(wn, diff_t, color = 'palevioletred')
plt.title('Förändring av spektral emittans vid ökning av temperatur med 1K')
plt.ylabel('Spektral emittans [W/(m^2 Hz)]')
plt.xlabel('Vågtal [cm-1]')
#plt.savefig('emittans+1K.png', dpi = 300)

#%%

svar_25 = ekv_2(f, t_ny, vmr)
olr_25 = ekv_3(svar_25, f)
olr_t_diff = olr_25 - E
print(olr_t_diff)

#%% 

vmr_co2 = perturb_vmr(vmr, 1, 0.35, t)
diff_vmr_co2 = ekv_2(f, t, vmr_co2) - ekv_2(f, t, vmr)
plt.plot(wn, diff_vmr_co2, color = 'palevioletred')
plt.title('Förändring av spektral emittans vid 35% ökning av CO2')
plt.ylabel('Spektral emittans /(m^2 Hz)]')
plt.xlabel('Vågtal [cm-1]')
#plt.savefig('emittans35CO2.png', dpi = 300)

#%%

svar_27 = ekv_2(f, t, vmr_co2)
olr_27 = ekv_3(svar_27, f)
olr_co2_diff = olr_27 - E
print(olr_co2_diff)

#%%

vmr_o3 = perturb_vmr(vmr, 2, 0.3, t)
diff_vmr_o3 = ekv_2(f, t, vmr_o3) - ekv_2(f, t, vmr)
plt.plot(wn, diff_vmr_o3, color = 'palevioletred')
plt.title('Förändring av spektral emittans vid 30% ökning av O3')
plt.ylabel('Spektral emittans [W/(m^2 Hz)]')
plt.xlabel('Vågtal [cm-1]')
#plt.savefig('emittans30O3.png', dpi = 300)

#%%

svar_30 = ekv_2(f, t, vmr_o3)
olr_30 = ekv_3(svar_30, f)
olr_o3_diff = olr_30 - E
print(olr_o3_diff)
#%%

vmr_ch4 = perturb_vmr(vmr, 3, 1.5, t)
diff_vmr_ch4 = ekv_2(f, t, vmr_ch4) - ekv_2(f, t, vmr)
plt.plot(wn, diff_vmr_ch4, color = 'palevioletred')
plt.title('Förändring av spektral emittans vid 150% ökning av CH4')
plt.ylabel('Spektral emittans [W/(m^2 Hz)]')
plt.xlabel('Vågtal [cm-1]')
#plt.savefig('emittans150CH4.png', dpi = 300)

#%%

svar_33 = ekv_2(f, t, vmr_ch4)
olr_33 = ekv_3(svar_33, f)
olr_ch4_diff = olr_33 - E
print(olr_ch4_diff)

#%%

vmr_n2o = perturb_vmr(vmr, 4, 0.2, t)
diff_vmr_n2o = ekv_2(f, t, vmr_n2o) - ekv_2(f, t, vmr)
plt.plot(wn, diff_vmr_n2o, color = 'palevioletred')
plt.title('Förändring av spektral emittans vid 20% ökning av N2O')
plt.ylabel('Spektral emittans [W/(m^2 Hz)]')
plt.xlabel('Vågtal [cm-1]')
#plt.savefig('emittans20N2O.png', dpi = 300)

#%% 

svar_36 = ekv_2(f, t, vmr_n2o)
olr_36 = ekv_3(svar_36, f)
olr_n2o_diff = olr_36 - E
print(olr_n2o_diff)

#%%

vmr_h2o = perturb_vmr(vmr, 0, 0.095, t)
svar_38 = ekv_2(f, t, vmr_h2o)
olr_38 = ekv_3(svar_38, f)
olr_h2o_diff = olr_38 - E
print(olr_h2o_diff)

#%%

vmr_co2_dubbel = perturb_vmr(vmr, 1, 1, t)
svar_40 = ekv_2(f, t, vmr_co2_dubbel)
olr_40 = ekv_3(svar_40, f)
olr_co2_dubbel_diff = olr_40 - E
print(olr_co2_dubbel_diff)