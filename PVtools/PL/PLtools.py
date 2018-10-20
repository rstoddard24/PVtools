import numpy as np
import math
import scipy
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.integrate import quad

#Constants
pi = math.pi
heV = 4.14e-15 #eV*s
c = 2.99792e8 #m/s
kbeV = 8.6173e-5 #eV/K
keV = 8.6173e-5 #eV/K
h = 6.626e-34
kb = 1.38065e-23
q = 1.60218e-19



#This module contains functions for Photoluminescence data analysis and modeling

def aipl(data,dark,grating):
    """
    This function takes PL data in cts/second units and
    converts to AIPL based on a laser power and grating calibration
    file. Functionality is built in to handle both single and map files

    INPUTS:
    data - data matrix containing input wavelength and PL cts/sec data
           if m x 2 matrix, treats as single spectra file
           if m x n matrix, treats as map along m
           if n x m matrix, treats as map along n

    dark - can be 0

    grating - specifies which grating used, a string either '500nm' or '1200nm'
    or '1200nm-InGaAs'
    
    OUTPUTS:
    aipl_data - data converted to absolute units , [=] photons/m^2-s-eV
    """
    
    #Get grating calibration file, then calculate conversion factor
    def BBPhotonFluxPerNM(lam,T):
        a = 2*pi/(h**3*c**2)*((h*c/(lam*1e-9))**2/(np.exp((h*c/(lam*1e-9))/(kb*T))-1))*(h*c/(lam*1e-9)**2)*1e-9
        return a
    if grating == '500nm':
        BB1050 = np.loadtxt('../../data/PLdata/grating_calibration_files/150 500'
                    'blaze BB files/BB 1050 10 um hole 10x SiCCD 532 LP'
                    'F No Duoscan Autoscanning_2.txt')

        BB_raw_photon_data = BB1050[:,1]/np.insert(BB1050[1:,0]-BB1050[:-1,0], 
                                           0,BB1050[1,0]-BB1050[0,0])

        

        AbsFluxesPerNM = np.zeros(BB1050.shape[0])
        Ts = 1050;

        for ii in range(BB1050.shape[0]):
            AbsFluxesPerNM[ii] = BBPhotonFluxPerNM(BB1050[ii,0],Ts+273.15)

        AbsPhotonRate = pi*(10/2*1e-6)**2*AbsFluxesPerNM #photons/sec-nm
        Conversion_factor = AbsPhotonRate/BB_raw_photon_data

        Ave_conv_factors = np.zeros([BB1050.shape[0],2])
        Ave_conv_factors[:,0] = BB1050[:,0]
        Ave_conv_factors[:,1] = Conversion_factor
        f2 = interp1d(Ave_conv_factors[:,0], Ave_conv_factors[:,1], kind='cubic')
    elif grating == '1200nm': 
        BB850 = np.loadtxt('../../data/PLdata/grating_calibration_files/BB 850C 10 um hole D0 10x 150 grating CCD 532 nm NoDS.txt')
        BB950 = np.loadtxt('../../data/PLdata/grating_calibration_files/BB 950C 10 um hole D0 10x 150 grating CCD 532 nm NoDS.txt')
        BB1050 = np.loadtxt('../../data/PLdata/grating_calibration_files/BB 1050C 10 um hole D0 10x 150 grating CCD 532 nm NoDS.txt')

        BB_raw_photon_data_1 = BB850[:,1]/np.insert(BB1050[1:,0]-BB1050[:-1,0], 
                                           0,BB1050[1,0]-BB1050[0,0])
        BB_raw_photon_data_2 = BB950[:,1]/np.insert(BB1050[1:,0]-BB1050[:-1,0], 
                                           0,BB1050[1,0]-BB1050[0,0])
        BB_raw_photon_data_3 = BB1050[:,1]/np.insert(BB1050[1:,0]-BB1050[:-1,0], 
                                           0,BB1050[1,0]-BB1050[0,0])
        
        BB_raw_photon_data = np.array([BB_raw_photon_data_1,BB_raw_photon_data_2,BB_raw_photon_data_3])
        
        AbsFluxesPerNM = np.zeros(BB_raw_photon_data.shape)
        for lam in range(len(BB_raw_photon_data_1)):
            tt = 0
            for T in (850,950,1050):
                AbsFluxesPerNM[tt,lam] = BBPhotonFluxPerNM(BB850[lam,0],T+273.15)
                tt += 1
        
        AbsPhotonRate = pi*(10/2*1e-6)**2*AbsFluxesPerNM #photons/sec-nm
        Conversion_factor = AbsPhotonRate/BB_raw_photon_data
        
        Ave_conv_factors = np.zeros([BB850.shape[0],2])
        Ave_conv_factors[:,0] = BB850[:,0]
        Ave_conv_factors[:,1] = np.mean(Conversion_factor,0)
        f2 = interp1d(Ave_conv_factors[:,0], Ave_conv_factors[:,1], kind='cubic')
    elif grating == '1200nm-InGaAs': 
        BB850 = np.loadtxt('../../data/PLdata/grating_calibration_files/Response_Synapse CCD2_784_150_Objective_x10_UV_0_Detector_Second_InjRej_Edge 785nm PL.txt')

        BB_raw_photon_data = BB850[:,1]/np.insert(BB850[1:,0]-BB850[:-1,0], 
                                           0,BB850[1,0]-BB850[0,0])

        

        AbsFluxesPerNM = np.zeros(BB850.shape[0])
        Ts = 850;

        for ii in range(BB850.shape[0]):
            AbsFluxesPerNM[ii] = BBPhotonFluxPerNM(BB850[ii,0],Ts+273.15)

        AbsPhotonRate = pi*(10/2*1e-6)**2*AbsFluxesPerNM #photons/sec-nm
        Conversion_factor = AbsPhotonRate/BB_raw_photon_data

        Ave_conv_factors = np.zeros([BB850.shape[0],2])
        Ave_conv_factors[:,0] = BB850[:,0]
        Ave_conv_factors[:,1] = Conversion_factor
        f2 = interp1d(Ave_conv_factors[:,0], Ave_conv_factors[:,1], kind='cubic')
    if data.shape[1] == 2: #single spectrum
        aipl_data = data    
        lam = data[:,0]
        Ipl_raw = data[:,1] #cts/sec
        if dark == []:
            Ipl_raw2 = Ipl_raw
        else:
            Ipl_raw = Ipl_raw - dark[:,1]
        Ipl_raw2 = Ipl_raw/np.insert(lam[1:]-lam[:-1],0,lam[1]-lam[0]) #cts/sec-nm    
        Ipl_nm = Ipl_raw2*f2(lam) #photons/sec-nm
        bandwidth_conv = np.insert(lam[1:]-lam[:-1],0,lam[1]-lam[0])/(heV*c/(lam*1e-9)**2*np.insert(lam[1:]-lam[:-1],0,lam[1]-lam[0])*1e-9)
        Ipl = Ipl_nm*bandwidth_conv/(pi*(6.01e-6)**2*2*0.921) #photons/sec-eV-m^2 (divide by factor of 2 since only considering FWHM beam area) (divide by 0.921 for window)
        aipl_data[:,1] = Ipl
    else:
        aipl_data = data
        k = 0
        while np.isnan(data[0,k]):
            k = k + 1
        lam = data[0,k:]
        for ii in range(1,data.shape[0]):
            Ipl_raw = data[ii,k:]
            if dark == []:
                Ipl_raw2 = Ipl_raw
            else:
                Ipl_raw = Ipl_raw - dark[:,1]
            Ipl_raw2 = Ipl_raw/np.insert(lam[1:]-lam[:-1],0,lam[1]-lam[0]) #cts/sec-nm    
            Ipl_nm = Ipl_raw2*f2(lam) #photons/sec-nm
            bandwidth_conv = np.insert(lam[1:]-lam[:-1],0,lam[1]-lam[0])/(heV*c/(lam*1e-9)**2*np.insert(lam[1:]-lam[:-1],0,lam[1]-lam[0])*1e-9)
            Ipl = Ipl_nm*bandwidth_conv/(pi*(6.01e-6)**2*2*0.921) #photons/sec-eV-m^2 (divide by factor of 2 since only considering FWHM beam area) (divide by 0.921 for window)
            aipl_data[ii,k:] = Ipl
    return aipl_data
    

def plqy_ext(aipl_data,laser_power):
    DiodeReadings_1sun = laser_power
    DiodeResponse532= 0.2741
    Area785ImageJ = pi*(6.01e-6)**2
    Ep532 = 2.3305 #E per photon @532

    #Load data from Mathmatica calcs to determine SQ limits @ 300 K and 350 K for various
    #Egs
    Egs = np.loadtxt('../../data/PLdata/vocmax_data/Egs.txt')
    VocSQs300 = np.loadtxt('../../data/PLdata/vocmax_data/VocMaxs.txt') # 300 K
    Jphs = np.loadtxt('../../data/PLdata/vocmax_data/Jphs.txt') #300 K
    VocSQs350 = np.loadtxt('../../data/PLdata/vocmax_data/VocMaxs2.txt') # 350 K
    VocSQs300_fn = interp1d(Egs, VocSQs300, kind='cubic')
    VocSQs350_fn = interp1d(Egs, VocSQs350, kind='cubic')
    Jphs_fn = interp1d(Egs, Jphs, kind='cubic')


    DiodeReading = DiodeReadings_1sun
    P532 = DiodeReading/(DiodeResponse532*Area785ImageJ*10) #W/m^2
    Jp532 = DiodeReading*0.925/(DiodeResponse532*Area785ImageJ*1.60218e-19*Ep532*2)

    if aipl_data.shape[1] == 2: #single spectrum
        lam = aipl_data[:,0]
        E = heV*c/(lam*1e-9)
        Ipl = aipl_data[:,1] 
        maxI = np.max(Ipl)
        maxI_idx = np.argmax(Ipl)
        peak_pos = E[maxI_idx]
        HHMax_idx = np.argmin(np.absolute(maxI/2-Ipl[:maxI_idx]))
        LHMax_idx = np.argmin(np.absolute(maxI/2-Ipl[maxI_idx:]))
        LHMax_idx = LHMax_idx+maxI_idx-1
        FWHM = E[HHMax_idx]-E[LHMax_idx]
        VocSQ300 = VocSQs300_fn(E[maxI_idx])
        VocSQ350 = VocSQs350_fn(E[maxI_idx])    
        JphSQ = Jphs_fn(E[maxI_idx])
        NSuns = Jp532*q/JphSQ;
        VocMax300 = VocSQ300 + kb*300/q*np.log(Jp532*q/JphSQ)
        VocMax350 = VocSQ350 + kb*350/q*np.log(Jp532*q/JphSQ)
        TotalPL = np.mean(-E[1:-1]+E[0:-2])/2*(Ipl[0]+Ipl[-1]+2*np.sum(Ipl[1:-2]))
        TotalPL_Eg = np.mean(-E[1:maxI_idx]+E[0:maxI_idx-1])/2*(Ipl[0]+Ipl[maxI_idx]+2*np.sum(Ipl[1:maxI_idx-1]))
        PLQY = TotalPL/Jp532
        dmu_PLQY = VocMax350-kbeV*350*np.log(1/PLQY)
        chi_PLQY = dmu_PLQY/VocMax300 
        chi_PLQY_Eg = (VocMax350-kbeV*350*np.log(1/(TotalPL_Eg/Jp532)))/VocMax300
        PLQY_Eg = TotalPL_Eg/Jp532
        dmu_PLQY_Eg = VocMax350-kbeV*350*np.log(1/(TotalPL_Eg/Jp532))
        mean_Ipl = np.sum(Ipl*E)/np.sum(Ipl)
    else: #maps
        k = 0
        while np.isnan(aipl_data[0,k]):
            k = k + 1
        lam = aipl_data[0,k:]
        E = heV*c/(lam*1e-9)
        mean_Ipl = np.zeros(aipl_data.shape[0]-1)
        peak_pos = np.zeros(aipl_data.shape[0]-1)
        FWHM = np.zeros(aipl_data.shape[0]-1)
        PLQY = np.zeros(aipl_data.shape[0]-1)
        dmu_PLQY = np.zeros(aipl_data.shape[0]-1)
        chi_PLQY = np.zeros(aipl_data.shape[0]-1)
        dmu_PLQY_Eg = np.zeros(aipl_data.shape[0]-1)
        chi_PLQY_Eg = np.zeros(aipl_data.shape[0]-1)
        for ii in range(1,aipl_data.shape[0]):
            Ipl = aipl_data[ii,k:]
            maxI = np.max(Ipl)
            maxI_idx = np.argmax(Ipl)
            peak_pos[ii-1] = E[maxI_idx]
            HHMax_idx = np.argmin(np.absolute(maxI/2-Ipl[:maxI_idx]))
            LHMax_idx = np.argmin(np.absolute(maxI/2-Ipl[maxI_idx:]))
            LHMax_idx = LHMax_idx+maxI_idx-1
            FWHM[ii-1] = E[HHMax_idx]-E[LHMax_idx]
            try:
                VocSQ300 = VocSQs300_fn(E[maxI_idx])
                VocSQ350 = VocSQs350_fn(E[maxI_idx])    
                JphSQ = Jphs_fn(E[maxI_idx])
                NSuns = Jp532*q/JphSQ;
                VocMax300 = VocSQ300 + kb*300/q*np.log(Jp532*q/JphSQ)
                VocMax350 = VocSQ350 + kb*350/q*np.log(Jp532*q/JphSQ)
                TotalPL = np.mean(-E[1:-1]+E[0:-2])/2*(Ipl[0]+Ipl[-1]+2*np.sum(Ipl[1:-2]))
                TotalPL_Eg = np.mean(-E[1:maxI_idx]+E[0:maxI_idx-1])/2*(Ipl[0]+Ipl[maxI_idx]+2*np.sum(Ipl[1:maxI_idx-1]))
                PLQY[ii-1] = TotalPL/Jp532
                dmu_PLQY[ii-1] = VocMax350-kbeV*350*np.log(1/PLQY[ii-1])
                chi_PLQY[ii-1] = dmu_PLQY[ii-1]/VocMax300 
                chi_PLQY_Eg[ii-1] = (VocMax350-kbeV*350*np.log(1/(TotalPL_Eg/Jp532)))/VocMax300
                PLQY_Eg = TotalPL_Eg/Jp532
                dmu_PLQY_Eg[ii-1] = VocMax350-kbeV*350*np.log(1/(TotalPL_Eg/Jp532))
                mean_Ipl[ii-1] = np.sum(Ipl*E)/np.sum(Ipl)
            except ValueError:
                VocSQ300 = 0
                VocSQ350 = 0
                JphSQ = 0
                NSuns = 1     
    return (mean_Ipl,peak_pos,FWHM,PLQY,dmu_PLQY,chi_PLQY,dmu_PLQY_Eg,chi_PLQY_Eg)

def med_idx(aipl_data):
    '''
    This function finds the index the AIPL spectrum with median PLQY
    '''
    k = 0
    while np.isnan(aipl_data[0,k]):
        k = k + 1
    lam = aipl_data[0,k:]
    E = heV*c/(lam*1e-9)
    TotalPL = np.zeros(aipl_data.shape[0]-1)
    for ii in range(1,aipl_data.shape[0]):
        Ipl = aipl_data[ii,k:]
        TotalPL[ii-1] = np.mean(-E[1:-1]+E[0:-2])/2*(Ipl[0]+Ipl[-1]+2*np.sum(Ipl[1:-2]))
    idx = np.argsort(TotalPL)[len(TotalPL)//2]
    return (idx+1)

def LSWK(E,theta,gam,Eg,QFLS,T):
    '''
    The Lasher-Stern-Wuerfel-Katahara equation
    '''

    '''
    theta = X[0]
    gam = X[1]
    #a0 = X(3)*Xscale(3);
    a0 = X[2]
    Eg = X[3]
    #Eg = Xscale(4);
    QFLS = X[4]
    T = X[5]
    #T = Xscale(6);
    d = 375/(1e7)
    '''
    a0 = 1e5
    d = 375/(1e7)
    ge = np.zeros(E.shape[0])

    for ii in range(E.shape[0]):
        ge[ii] = 1/(gam*2*scipy.special.gamma(1+1/theta))*quad(lambda u: np.exp(-np.absolute(u/gam)**theta)*np.sqrt((E[ii]-Eg)-u),-math.inf,E[ii]-Eg)[0]

    AIPL = 2*pi*E**2/(heV**3*c**2)*((1-np.exp(-a0*d*ge))/(np.exp((E-QFLS)/(keV*T))-1))*(1-2/(np.exp((E-QFLS)/(2*keV*T))+1))

    AIPL = np.log(AIPL)
    return AIPL

"""
#Load GFuncTable and make interpolation function to speed up full peak fit
g_func_table = np.loadtxt('../../data/PLdata/GFuncTables/GFuncTable.csv',delimiter=',')
g_interp_func = interp2d(g_func_table[:,0],g_func_table[:,1],g_func_table[:,2])
def LSWK_gfunc(E,theta,gam,Eg,QFLS,T):
    '''
    The Lasher-Stern-Wuerfel-Katahara equation
    '''

    '''
    theta = X[0]
    gam = X[1]
    #a0 = X(3)*Xscale(3);
    a0 = X[2]
    Eg = X[3]
    #Eg = Xscale(4);
    QFLS = X[4]
    T = X[5]
    #T = Xscale(6);
    d = 375/(1e7)
    '''
    a0 = 1e5
    d = 375/(1e7)
    
    
    ge = np.zeros(E.shape[0])

    for ii in range(E.shape[0]):
        ge[ii] = g_interp_func(theta, (E[ii]-Eg)/gam)
    
    AIPL = 2*pi*E**2/(heV**3*c**2)*((1-np.exp(-a0*d*ge))/(np.exp((E-QFLS)/(keV*T))-1))*(1-2/(np.exp((E-QFLS)/(2*keV*T))+1))

    AIPL = np.log(AIPL)
    return AIPL

"""
def full_peak_fit(E,Ipl):
    thresh = 5e18
    maxI_idx = np.argmax(Ipl)
    lb_idx = np.argmin(np.absolute(Ipl[:maxI_idx]-thresh))
    rb_idx = np.argmin(np.absolute(Ipl[maxI_idx:]-thresh))+maxI_idx
    
    #ll_idx = np.argmin(np.absolute(E-1.7))
    
    X = np.ones(5);
    d = 375/(1e7) #cm
    Xscale = [1.5,.037,1e5,1.75,1.4,288,d]
    X[0] = Xscale[0]
    X[1] = Xscale[1]
    #X[2] = Xscale[2]
    X[2] = Xscale[3]
    X[3] = Xscale[4]
    X[4] = Xscale[5]
    
    
    (Xf, pcov) = curve_fit(LSWK, E[lb_idx:rb_idx], np.log(Ipl[lb_idx:rb_idx]),p0=X)
   

    #aipl_mod = fpf(E,Xf[0],Xf[1],Xf[2],Xf[3],Xf[4])
    aipl_mod = np.exp(LSWK(E[lb_idx:rb_idx],Xf[0],Xf[1],Xf[2],Xf[3],Xf[4]))
    return (E[lb_idx:rb_idx], aipl_mod,Xf[0],Xf[1],Xf[2],Xf[3],Xf[4])
