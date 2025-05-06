import numpy as np
import math
import pandas as p
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from scipy.stats import linregress
from matplotlib.ticker import ScalarFormatter
import scipy.optimize as opt
import seaborn as sns
from sklearn.metrics import r2_score
import matplotlib.lines as mlines
from matplotlib.lines import Line2D




# %% 1 - FIGURE INITIALISATION

# Function for establishing consistent figures
# Set optional logOption arg to true for log scale graphs. Must contain x and y data to construct axis tick points
def figSettings(logOption = False, x_data = None, y_data = None, y_errors = None):
    
    # General font and figure settings
    plt.rcParams.update({"text.usetex": False,
                         "font.family": "sans-serif",
                         "font.sans-serif": "Arial",
                         "font.size": 7,
                         "axes.labelsize": 7,
                         "legend.fontsize": 6,
                         "xtick.labelsize": 6,
                         "ytick.labelsize": 6,
                         "figure.figsize": (3.5,2.5),
                         "figure.dpi": 900,
                         "savefig.dpi": 900,
                         "legend.frameon": False,
                         "axes.linewidth": 0.9,
                         "mathtext.fontset": 'stixsans'})
    
    # Initialise figure and axes
    fig, ax = plt.subplots()
    
    # Set ticks to go into the figure, specify length and width, have ticks on all sides, and have no margin padding
    ax.xaxis.set_tick_params(direction='in', length=2.5, width= 1)
    ax.yaxis.set_tick_params(direction='in', length=2.5, width = 1)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.margins(0)
    
    
    # If a log graph is selected and data has been entered for bounding
    if logOption and x_data is not None and y_data is not None:
        
        # Set axis to log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Set minor ticks in between base-10 major ticks to have 10 per major tick
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=10))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=10))
        
        # Set minor ticks to go into the page and set size
        ax.tick_params(axis='both', which='minor', direction='in', length=1.5, width= 0.75)
        
        # Find the closest base-10 increment to the maximum and minimum of the x and y data
        x_min = 10 ** np.floor(np.log10(np.min(x_data)))
        x_max = 10 ** np.ceil(np.log10(np.max(x_data)))

        # Set axis limits to these closest points, thus starting and ending the axis on a major tick point
        ax.set_xlim([x_min, x_max])
        
        
        if y_errors is not None:
            
            all_ymin = []
            all_ymax = []
            
            for y, err in zip(y_data, y_errors):
                
                y = np.array(y)
                err = np.array(err)
                y_lower = y - err
                y_upper = y + err
                
                all_ymin.append(np.min(y_lower))
                all_ymax.append(np.max(y_upper))
                
            y_min_val = np.min([v for v in all_ymin if v > 0])
            y_max_val = np.max(all_ymax)
            
        else:
            
            if isinstance(y_data,np.ndarray):
                all_y_values = y_data
            else:
                all_y_values = np.concatenate(y_data)
            y_min_val = np.min(all_y_values[all_y_values > 0])
            y_max_val = np.max(all_y_values)
            
        y_min = 10 ** np.floor(np.log10(y_min_val))
        y_max = 10 ** np.ceil(np.log10(y_max_val))
        
        ax.set_ylim([y_min, y_max])
        
    else:
        if x_data is not None and y_data is not None:
            # Estimate tick spacing using AutoLocator
            x_locator = ticker.AutoLocator()
            y_locator = ticker.AutoLocator()
            
            # Get tick step from dummy ticks
            x_tick_spacing = np.diff(x_locator.tick_values(np.min(x_data), np.max(x_data)))[0]
            y_tick_spacing = np.diff(y_locator.tick_values(np.min(y_data), np.max(y_data)))[0]
            
            x_min = np.floor(np.min(x_data) / x_tick_spacing) * x_tick_spacing
            x_max = np.ceil(np.max(x_data) / x_tick_spacing) * x_tick_spacing
            y_min = np.floor(np.min(y_data) / y_tick_spacing) * y_tick_spacing
            y_max = np.ceil(np.max(y_data) / y_tick_spacing) * y_tick_spacing
            
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            
            ax.xaxis.set_major_locator(x_locator)
            ax.yaxis.set_major_locator(y_locator)
                     
    return fig, ax

# %% 2 - VELOCITY AND STRESS

# %%% 2.1 Stress
def stressCalc(r, slipNumber):
    
    # Calculating the force on the plate using know coverslip properties
    mass = 0.16504 * slipNumber # grams
    diameter = 24e-3 # meters
    thickness = 0.145e-3 # meters
    g = 9.81 # m/s/s
    rho = 950 # kg/m3, density of silicone oil
    F_w = (mass/1000)*g # weight of the plate
    F_b = math.pi*((diameter/2)**2)*thickness*rho*g*slipNumber # buouyancy force of the plate
    F_n = F_w-F_b # total force of the plate

    # Calculate the normal stress applied to the sphere at each moment by = F/A
    stress = F_n/(math.pi * (r**2))
    
    return stress

# %%% 2.2 Velocity 
# Function to calculate the velocity of sphere expansion
def velocCalc(sheetName, folderName, vidLength, slipNumber, trim,  n = 1):
    
    if debug:
        print(sheetName)
    
    # Add csv to complete filename
    fileName = sheetName +'.csv'
    
    # Extract radius from ImageJ area profile file
    r = radiusExtract(folderName, fileName)
    
    # Find the length of the array (i.e. number of frames) and make a time array of the same length up to the video length
    fLen = len(r)
    t = np.linspace(0,vidLength,fLen)
    
    # Create a boolean mask that is True for only values within the specified trimming/cropping
    mask = (t >= trim[0]) & (t <= trim[1])
    
    # Trim the radius and time arrays to only the time of interest
    r = r[mask]
    t = t[mask] 
    
    # Downsample both time and radius due to irregular sampling of measurements by imageJ
    # Allows a smoother function for analysis
    samp_r = r[::n]
    samp_t = t[::n] 

    # Differentiate the radius with respect to time/frames, i.e. find the velocity !!!
    veloc = np.gradient(samp_r)

    # Calculate the normal stress applied to the sphere at each moment
    stress = stressCalc(samp_r, slipNumber)
    
    return veloc, stress, samp_r, samp_t

# %% 3 - LINEAR REGRESSION FITTING

# %%% 3.1 Log-log Fit
# Function for log log linear regression fit
def logFit(x, y):
    
    # Find logarithms
    logX = np.log(x)
    logY = np.log(y)
    
    # Find the linear line of best fit for these data
    linOut = linregress(logX, logY)
    
    # Sort output, gradient (m), intercept (c), PMCC squared (r2)
    m = linOut[0]
    c = linOut[1]
    r2 = linOut[2]**2
    
    return m, c, r2

# %%% 3.2 Square root fit
# Function for square root - square root linear regression fit
def sqrtFit(x, y):
    
    # Find square roots 
    sX = np.sqrt(x)
    sY = np.sqrt(y)
    
    # Find the linear line of best fit for these data
    linOut = linregress(sX, sY)
    
    # Sort output, gradient (m), intercept (c), PMCC squared (r2)
    m = linOut[0]
    c = linOut[1]
    r2 = linOut[2]**2
    
    return m, c, r2

# %%% 3.3 Linear Regression through origin
# Function for finding the line of best fit that passes through the origin (for Young's modulus)
def linearFitThroughZero(x, y):
    
    # Make numpy arrays for easier array manipulation
    x = np.array(x)
    y = np.array(y)
    
    # Find the gradient that satisfies the sum of the least squares
    m = np.sum(x*y)/np.sum(x**2)
    
    return m

# %%% 3.4 R2 Value Calculation

def r2Calc(y_true, y_pred):
    s_residual = np.sum((y_true - y_pred) ** 2)
    s_total = np.sum(y_true ** 2)
    
    r2 = 1 - s_residual / s_total
    
    return r2

# %% 4 - STRAIN

# %%% 4.2 Dyanmic Strain and Shear Rate (for Viscosity)
# Function for calculating the penetration depth and strain of the spheres using !!! method
def strainCalcDyn(rI, rF, start = 1e-4):
    
    # Find the length of the input radii array and initialise penetration depth (delta), strain (e), and shear rate (shear) arrays
    length = len(rF)
    delta = [0]*length
    e = [0]*length
    shear = [0]*(length-1)
    
    # For every radius
    for i in range(length):
    
        # Reset the iterative calculation difference on each strain calculation and use the standard starting point
        diff = 1 
        d = start
    
        # While the difference between the newly calculated penetration depth and previous is still signficant: keep calculating!
        # I.e. not yet converged
        while diff > 1e-9:
            
            # Iterative formula (delta on both sides). Uses the conservation of momentum from !!!
            d_n = rI - ((1/(3*(rF[i]**2))) * ((2*(rI**3)) + (rI - d)**3))
            
            # Absolute difference between old and new deltas
            diff = abs(d_n - d)
            
            # Make the new delta the old one for the next loop
            d = d_n

        # Once convergence is attained the final delta value is equal to this radius's delta            
        delta[i] = d
        
        # Strain is penetration depth/initial radius
        e[i] = delta[i]/rI
        
    # Differentiate the strain (by time) to find the shear rate in /s
    shear = np.gradient(e)
        
    return delta, e, shear

# %%% 4.3 Static Strain (for Young's Modulus)

# Function for calculating the penetration depth and strain of the spheres using !!! method
def strainCalcStatic(rI, rF, start = 1E-4):
    
    diff = 1
    d = start
    
    while diff > 1e-9:
        
        d_n = rI - ((1/(3*(rF**2))) * ((2*(rI**3)) + (rI - d)**3))
        
        diff = abs(d_n - d)
        
        d = d_n
        
    delta = d
    
    e = delta/rI
    
    return e, delta

# %%% 4.4 Volumetric Strain Calculation

    


# %% 5 - RHEOLOGICAL MODELS    

# %%% 5.1 Power Law
# Function to find the power law stress-shear rate curve
def powLaw(shear, stress, debug=True):
    
    # Fit a log log fit to the shear rate and stress, n = m, K = e^c
    m, c, r2 = logFit(shear, stress)
    K = math.exp(c)
    
    # If debug is true then print power law parameters into log
    if debug:
        print('Power law fit: n = ' +str(m)+'; K = ' +str(K)+ '; R2 = ' +str(r2))
    
    return m, K, r2

# %%% 5.2 Hershel Buckley

# Function for the Hershel-Buckley equation. For use in the multivariable optimisation algorithm
def hershelObjective(yieldStress, shear, stress):
    
    # Equation only valid when stress exceeds the yield stress, thus take a boolean mask to filter unviable options
    valid_indices = stress > yieldStress
    
    # If only 1 or no stresses are valid, return infinity (break optimisation)
    if np.sum(valid_indices) < 2:
        return np.inf
    
    # Find the log of both sides log(tau - tau_y) = nlog(gamma) + logK
    logShear = np.log(shear[valid_indices])
    logStress = np.log(stress[valid_indices] - yieldStress)
    
    # Perform linear regression and find the line of the best fit parameters
    m, c, r, _, _ = linregress(logShear, logStress)
    
    return -r**2


def HershelBuckley(shear, stress, debug=True):
    
    # Find optimum yield stress
    result = opt.minimize_scalar(hershelObjective, args=(shear, stress), bounds=(0, min(stress)-1e-6), method ='bounded')
    yieldStress = result.x
    
    # Use yield stress to find hershel fit
    valid_indices = stress > yieldStress
    stress = stress[valid_indices] - yieldStress
    shear = shear[valid_indices]
    
    
    m, c, r2 = logFit(shear, stress)
    K = np.exp(c)
    
    if debug:
        print('Hershel fit: n = ' +str(m)+'; K = ' +str(K)+ '; Yield Stress = ' +str(yieldStress) + '; R2 = ' +str(r2))
    
    return m, K, yieldStress, r2

# %%% 5.3 Casson

def Casson(shear, stress, debug=True):
    
    m, c, r2 = sqrtFit(shear, stress)
    
    yieldStress = c**2
    K=m**2
    
    if debug:
        print('Casson fit: K = ' +str(K)+ '; Yield Stress = ' +str(yieldStress) + '; R2 = ' +str(r2))
    
    return K, yieldStress, r2

# %%% 5.4 Casson-Papanastasiou

def cassonPapanFunc(shear, mu_inf, ys, m):
    
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC9293974/
    # https://www.sciencedirect.com/science/article/pii/S0377025703000417?casa_token=f9_FvUj1rQsAAAAA:HI2DNNhx9DHUd60qC6WM_w3cbeBw2N0UmfNFMaNoQxKTqqFcmKVCMD2incMwAYuEPXioTq00MQ
    # mentions blood parameters within
    # https://www.academia.edu/6270049/Entry_and_exit_flows_of_casson_fluids
    # All three of these give different equations... annoying. This uses no.1 but no.3 looks most reputable?
    out = np.sqrt(mu_inf) + np.sqrt(ys/shear)*(1 - np.exp(-1* np.sqrt(m * shear)))
    
    
    return (out**2) * shear


def cassonPapan(shear, stress, debug=True, initGuess = [1, 50, 50]):
    
    # Avoid divide by 0
    mask = shear > 1e-8
    shear = shear[mask]
    stress = stress[mask]
    Bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
    
    params, _ = opt.curve_fit(cassonPapanFunc, shear, stress, p0= initGuess, bounds=Bounds, maxfev = 10000)

    mu_inf, tau_y, m = params
    
    stressPred = cassonPapanFunc(shear, mu_inf, tau_y, m)
    r2 = r2_score(stress, stressPred)
    
    if debug:
        print('Casson-Papanastasiou: mu_inf = ', mu_inf, '; Yield Stress = ', tau_y, '; m = ', m, '; R2 = ', r2)
    
    return mu_inf, tau_y, m, r2

# %%% 5.5 Hershel-Bulkley-Papanastasiou

def hershelPapanFunc(shear, n, K, m, ys):
    
    #https://www.sciencedirect.com/science/article/pii/S0377025712002315
    
    stress = (K*(shear**n)) + (ys*(1-np.exp(-m*shear)))

    return stress

def hershelPapan(shear, stress, debug=True, initGuess = [0.4, 5, 5, 10]):
    
    # Avoid divide by 0
    mask = shear > 1e-8
    shear = shear[mask]
    stress = stress[mask]
    Bounds=([0.1, 0, 0, 0], [1.5, 100, 100, 650]) # !!! SUPRESSING THIS OUTPUT AS IT IS OVERFITTING TOO MUCH
    
    params, _ = opt.curve_fit(hershelPapanFunc, shear, stress, p0 = initGuess, bounds = Bounds, maxfev= 10000)
    
    n, K, m, ys = params

    stressPred = hershelPapanFunc(shear, n, K, m, ys)
    r2 = r2_score(stress, stressPred)
    
    if debug:
        print('HB-Papanastasiou: n = ', n, '; Yield Stress = ', ys, '; m = ', m,'; K = ', K, '; R2 = ', r2)
    
    return n, K, m, ys, r2

# %%% 5.6 Young's Modulus Calculation
#https://www.nature.com/articles/s41598-019-49589-w#:~:text=The%20Poisson's%20ratios%2C%20reported%20by,%2C29%2C32%2C38.
#https://www.sciencedirect.com/science/article/pii/S1350453318300092?via=ihub#eqn0008 - Collagen type I Poissons ratio 0.29
def Youngs(G, v = 0.38):

    return 2*G*(1+v)

# %% 6 - MODEL TOOLS

# %%% 6.1 Radius ImageJ Extraction
def radiusExtract(folderName, fileName):
    
    # Initialise data array
    pData = {}
    
    # Retrieve area array of the specified experiment
    pData['Data'] = p.read_csv(folderName+fileName)
    A = np.array(pData['Data'].get('Area'))
    
    # Calculate the radius and divide by 1E6 as given in um
    r = np.sqrt(A/math.pi)/1000000
    
    return r

# %%% 6.2 Model Data Extraction and Assignment
def modelData_extraction(data, keys, mask=None):
    
    if mask is not None:
        
        return {key: np.array(data[key])[mask] for key in keys}
    
    return {key: np.array(data[key]) for key in keys}

# %%% 6.3 Rheometery Data Extraction
def rheometerData(matCode, StrainOnly = False):
    
    name = matSelector(matCode)
    
    if not StrainOnly:
        visc = p.read_excel("RheometerMASTER.xlsx", sheet_name=name)
        R_visc = np.array(visc.get('Avg'))
        R_visc = R_visc[1:]
        R_std = np.array(visc.get('Stdev'))
        R_std = R_std[1:]
        
        shear = p.read_excel("RheometerMASTER.xlsx", sheet_name= name+'_Strain')
        R_shear = np.array(shear.get('Avg'))
        R_shear = R_shear[1:]
        R_Sstd = np.array(shear.get('Stdev'))
        R_Sstd = R_Sstd[1:]
        
        return R_visc, R_std, R_shear, R_Sstd
    
    
    shear = p.read_excel("RheometerMASTER.xlsx", sheet_name= name+'_Strain')
    
    R_shear = np.array(shear.get('Avg'))
    R_shear = R_shear[1:]
    R_Sstd = np.array(shear.get('Stdev'))
    R_Sstd = R_Sstd[1:]

    return R_shear, R_Sstd


# %%% 6.4 Material Code Translator
def matSelector(matCode, shorten = False):
    
    if matCode == 'F' and not shorten:
        name = 'Fibrin'
    elif matCode == 'F' and shorten:
        name = 'F'
    elif matCode == 'C' and not shorten:
        name = 'Collagen'
    elif matCode == 'C' and shorten:
        name = 'C'
    elif matCode == 'H' and not shorten:
        name = 'Hybrid'
    elif matCode == 'H' and shorten:
        name = 'H'
    elif matCode == 'f' and not shorten:
        name = 'Fibrin Control'
    elif matCode == 'f' and shorten:
        name = 'FC'
    elif matCode == 'c' and not shorten:
        name = 'Collagen Control'
    elif matCode == 'c' and shorten:
        name = 'CC'
    elif matCode == 'L' and not shorten:
        name = 'Alginate-Calcium'
    elif matCode == 'L' and shorten:
        name = 'A-C'
    elif matCode == 'a' and not shorten:
        name = 'Agarose 0.5%'
    elif matCode == 'a' and shorten:
        name = 'A-0.5%'
    elif matCode == 'i' and not shorten:
        name = 'Fibrin Bioink - 3 Day'
    elif matCode == 'i' and shorten:
        name = 'FB-3'
    elif matCode == 'I' and not shorten:
        name = 'Fibrin Bioink - 7 Day'
    elif matCode == 'I' and shorten:
        name = 'FB-7'
    elif matCode == 'o' and not shorten:
        name = 'Collagen Bioink - 3 Day'
    elif matCode == 'o' and shorten:
        name = 'CB-3'
    elif matCode == 'O' and not shorten:
        name = 'Collagen Bioink - 7 Day'
    elif matCode == 'O' and shorten:
        name = 'CB-7'
    elif matCode == 'y' and not shorten:
        name = 'Hybrid Bioink - 3 Day'
    elif matCode == 'y' and shorten:
        name = 'HB-3'
    elif matCode == 'Y' and not shorten:
        name = 'Hybrid Bioink - 7 Day'
    elif matCode == 'Y' and shorten:
        name = 'HB-7'
        
    return name


# %%% 6.5 Model Name Selector
def modelSelector(modelHeading, shorten = False):
    
    if modelHeading == 'Power' and not shorten:
        name = 'Power Law'
    elif modelHeading == 'Power' and shorten:
        name = 'PL'
        
    elif modelHeading == 'Casson' and not shorten:
        name = 'Casson'
    elif modelHeading == 'Casson' and shorten:
        name = 'C'
        
    elif modelHeading == 'Cas-Papan' and not shorten:
        name = 'Casson-Papanastasiou'
    elif modelHeading == 'Cas-Papan' and shorten:
        name = 'C-P'
        
    elif modelHeading == 'Hershel' and not shorten:
        name = 'Herschel-Bulkley'
    elif modelHeading == 'Hershel' and shorten:
        name = 'HB'
        
    elif modelHeading == 'Her-Papan' and not shorten:
        name = 'Herschel-Bulkley-Papanastasiou'
    elif modelHeading == 'Her-Papan' and shorten:
        name = 'HB-P'
        
    else:
        name = modelHeading
    
    return name


# %% 7 - ERROR PROPAGATION

# %%% 7.1 Division Error Propogation
def errorPropogation(x, y, stdX, stdY, n_x, n_y):
    
    meanRatio = x/y
    
    SEM_x = stdX/math.sqrt(n_x)
    SEM_y = stdY/math.sqrt(n_y)
    
    stdRatio = meanRatio * np.sqrt(((SEM_x/x)**2) + ((SEM_y/y)**2))
    
    return meanRatio, stdRatio

# %%% 7.2  Standard Error Mean
def avgStdDev(std):
    
    # std_avg = np.sqrt(np.sum(std**2)) / len(std) # Standard error of the mean (SEM)
    std_avg = np.sqrt(np.mean(std**2)) # Pooled standard deviation

    return std_avg

# %%% 7.2 Mean Error Propagation
# Uses inverse-variance weighting to weight averages of standard deviations

def weighted_std(std, n):
    
    Std = np.array(std)
    N = np.array(n)
    
    std_avg = np.sqrt(np.sum((Std**2)/N)) / np.sum(1/N)
    
    return std_avg

# %%% 7.2 Pooled Standard Deviation

def pooledStd(std, n):
    
    Std = np.array(std)
    N = np.array(n)
    
    var_avg = np.sum((N - 1) * (Std**2)) / np.sum(N - 1)
    
    return np.sqrt(var_avg)
    

# %% 8 - MODEL PARAMATER CALCULATION

# %%% 8.1 Individual Experiment Calculation
def runFunction(sheet, folder, vidLength, slipNumber, trim, sampling, rI):
    
    # Find velocity and stress
    veloc, stress, samp_r, samp_t = velocCalc(sheet, folder, vidLength, slipNumber, trim, sampling)  
    
    # Find e and shear
    delta, e, shear = strainCalcDyn(rI, samp_r)
    
    # Power Law Fit
    pow_n, pow_K, pow_r2 = powLaw(shear, stress, debug)
    
    # Hershel Buckley Fit
    h_n, h_K, h_y , h_r2= HershelBuckley(shear, stress, debug)
    
    # Casson Fit
    c_K, c_y, c_r2 = Casson(shear, stress, debug)
    
    # Casson-Papanastasiou Fit
    cp_mu, cp_y, cp_m, cp_r2 = cassonPapan(shear, stress, debug)
    
    # Hershel-Buckley-Papanastasiou Fit
    hp_n, hp_K, hp_m, hp_ys, hp_r2 = hershelPapan(shear, stress, debug)
    
    return veloc, stress, samp_r, samp_t, delta, e, shear, pow_n, pow_K, pow_r2, h_n, h_K, h_y, h_r2, c_K, c_y, c_r2, cp_mu, cp_y, cp_m, cp_r2, hp_n, hp_K, hp_m, hp_ys, hp_r2



# %%% 8.2 Bulk Experiment Calculation
def bulkFunction():
    
    Data = {}
    Data['Inputs'] = p.read_excel('InputData-test2.xlsx', sheet_name='Sheet1')

    inputKeys = ['ID', 'Folder', 'Filename', 'SlipNo.', 'Start Radius', 'Vid Length (s)', 'Vid Start', 'Vid End', 'Sampling']
    
    inputs = modelData_extraction(Data['Inputs'], inputKeys)
    
    inputs['Sampling'] = inputs['Sampling'].astype(int)

    length = len(inputs['ID'])
    
   
    # Initialise 0 arrays for each model variable, named according to the keys defined globally in the script
    model_vars = {key: [0]*length for key in model_keys}
    
    # Loop for every single experiment and perform the run function on every one, taking the ourput and sorting into the correct keys
    for i in range(length):
        
        output = runFunction(inputs['Filename'][i], inputs['Folder'][i], inputs['Vid Length (s)'][i], inputs['SlipNo.'][i], 
                             [inputs['Vid Start'][i], inputs['Vid End'][i]], inputs['Sampling'][i], inputs['Start Radius'][i])

        # Loop for every key in the model variables dictionary, and pull the index and the key from there using enumerate, and assign the corresponding outputs 
        for j, key in enumerate(model_keys):
            
            # Add 7 to the output as the first 7 outputs of the run function are not of significance here
            model_vars[key][i] = output[j+7]
            
            
    
    columnHeadings = ['ID', 'Slip', 'Power n', 'Power K', 'Power R2', 'Hershel n', 'Hershel K', 'Hershel Yield', 'Hershel R2', 'Casson K', 'Casson Yield', 'Casson R2', 
                      'Cas-Papan Visc Limit', 'Cas-Papan Yield', 'Cas-Papan M', 'Cas-Papan R2', 
                      'Her-Papan n', 'Her-Papan K', 'Her-Papan M', 'Her-Papan Yield', 'Her-Papan R2']
    combined = list(zip(inputs['ID'], inputs['SlipNo.'], *[model_vars[k] for k in model_keys]))
    dataFrame = p.DataFrame(combined, columns=columnHeadings)

    dataFrame.to_excel('OutputData.xlsx', sheet_name='Sheet1')

    return combined, dataFrame


# %% 9 - VISCOSITY

# %%% 9.1 Effective Viscosity Protorheology Calculation
def allViscosity(dataFrame, matCode, shear):
    # matCode is the one letter code for the material. F = Fibrin, C = Collagen...

    keys = ['ID','Slip', 'Hershel n', 'Hershel K', 'Hershel Yield', 'Hershel R2',
          'Casson K', 'Casson Yield', 'Casson R2', 
          'Cas-Papan Visc Limit', 'Cas-Papan Yield', 'Cas-Papan M', 'Cas-Papan R2',
          'Her-Papan n', 'Her-Papan K', 'Her-Papan M', 'Her-Papan Yield', 'Her-Papan R2']
    
    # Create a boolean mask for the specified material using matCode, and using the previously defined function, remove irrelevant values
    mask = np.array([str(i)[0] == matCode for i in dataFrame['ID']])
    data = modelData_extraction(dataFrame, keys, mask)
    
    h_n, h_K, h_ys, h_r2 = data['Hershel n'], data['Hershel K'], data['Hershel Yield'], data['Hershel R2']
    c_K, c_ys, c_r2 = data['Casson K'], data['Casson Yield'], data['Casson R2']
    cp_mu, cp_ys, cp_m, cp_r2 = data['Cas-Papan Visc Limit'], data['Cas-Papan Yield'], data['Cas-Papan M'], data['Cas-Papan R2']
    hp_n, hp_K, hp_m, hp_ys, hp_r2 = data['Her-Papan n'], data['Her-Papan K'], data['Her-Papan M'], data['Her-Papan Yield'], data['Her-Papan R2']
    slip = data['Slip']

    avgVisc = {}
    stdVisc = {}
    nRepeats = {}
    ViscAll = []
    
    for s in np.unique(slip):
        
        # Initialise viscosity array
        visc = []
        
        # Boolean mask to consider only the slip of question
        slipMask = slip == s
        
                
        for H_r2, C_r2, CP_r2, HP_r2, H_ys, H_k, H_n, C_ys, C_k, CP_mu, CP_ys, CP_m, HP_n, HP_m, HP_K, HP_ys in zip(
                h_r2[slipMask], c_r2[slipMask], cp_r2[slipMask], hp_r2[slipMask], h_ys[slipMask], h_K[slipMask], h_n[slipMask], 
                c_ys[slipMask], c_K[slipMask], cp_mu[slipMask], cp_ys[slipMask], cp_m[slipMask], 
                hp_n[slipMask], hp_m[slipMask], hp_K[slipMask], hp_ys[slipMask]):
        
            R2vals = (H_r2, C_r2, CP_r2, HP_r2)
            maxR2 = R2vals.index(max(R2vals)) #index 0=HB, 1=Casson, 2=C-P, 3 HB-P
            
            if maxR2 == 0:        
        
                visc_val = (H_ys/shear) + (H_k*(shear**(H_n - 1)))
            
            elif maxR2 == 1:
                
                visc_val = (np.sqrt(C_ys/shear) + np.sqrt(C_k))**2
                
            elif maxR2 == 2:
                
                visc_val = (np.sqrt(CP_mu) + (np.sqrt(CP_ys/shear)*(1-np.exp(-1*np.sqrt(CP_m*shear)))))**2
            
            elif maxR2 == 3:
                
                visc_val = (HP_K*(shear**(HP_n-1))) + ((HP_ys/shear)*(1-np.exp(-1*HP_m*shear)))
            
            visc.append(visc_val)
            
            ViscAll.append(visc_val)


        avgVisc[s] = np.mean(visc, axis=0)
        stdVisc[s] = np.std(visc, axis=0)
        
        # Track the number of repeats occuring for proper error propogation (sample size)
        nRepeats[s] = np.array(visc).shape[0]
        
        All_avgVisc = np.mean(np.array(ViscAll), axis=0)
        All_stdVisc = np.std(np.array(ViscAll), axis=0)
        
        nSize = np.size(ViscAll, axis = 0)
        
        
        
    return avgVisc, stdVisc, nRepeats, All_avgVisc, All_stdVisc, nSize

# %%% 9.2 Protorheology to Rheology Viscosity Ratio    
def viscCompareData(dataFrame, matCode):
    
    # Pull rheometer data
    R_visc, R_std, R_shear, R_Sstd = rheometerData(matCode)

    # Get protorheology data
    avgVisc, stdVisc, nRepeats, All_avgVisc, All_stdVisc, nSize = allViscosity(dataFrame, matCode, R_shear)
    
    
    # Initialise dictionaries
    viscScale = {}
    error = {}
    
    # totalSlips = sorted(avgVisc.keys())    
    # print(totalSlips)
    
    n_y = 3 # Hard coded as rheometer always has 3 repeats 
    
    # for s in totalSlips:
        
    #     viscScale[s], error[s] = errorPropogation(All_avgVisc, R_visc, All_stdVisc, R_std, nSize, n_y)
    
    viscScale, error = errorPropogation(All_avgVisc, R_visc, All_stdVisc, R_std, nSize, n_y)
        
    
    return viscScale, error, R_shear, nRepeats


# %% 10 - FIGURE CREATION

# %%% 10.1 Manual, Single, Custom Graph Creation
def stdGraphMaker():

    # ------------------------------------------------------------------------------------------------------------------
    # PARAMETERS for velocity and stress calculation
    sheet = 'C-RH-0.75-1203_2102'
    folder = '0.75 Collagen FG 210225/'
    vidLength = 101.81
    slipNumber = 3
    trim = [20, 101.81]
    sampling = 30
    logScale = False # insert True here if log log scale wanted
    initial_r = 6.40E-04
    
    
    # GRAPH SELECTORS
    x_var = 'shear' # Graphable values: shear, strain, stress, radius, velocity, time
    y_var = 'stress' 
    # ------------------------------------------------------------------------------------------------------------------
    
    veloc, stress, samp_r, samp_t, delta, e, dudr, pow_n, pow_K, pow_r2, h_n, h_K, h_y, h_r2, c_K, c_y, c_r2, cp_mu, cp_y, cp_m, cp_r2, hp_n, hp_K, hp_m, hp_ys, hp_r2 = runFunction(sheet, folder, vidLength, slipNumber, trim, sampling, initial_r)
    
    
    fig,ax = figSettings(logScale) 
    
    if x_var == 'shear':
        x = dudr
        xlabel = r'Shear Rate, $\dot{\gamma}$ $(s^{-1})$'
    elif x_var == 'stress':
        x = stress
        xlabel = r'$\sigma$ $(Pa )$'
    elif x_var == 'radius':
        x = samp_r*1000000
        xlabel = r'$r$ $(\mu m)$'
    elif x_var == 'velocity':
        x = veloc*1000000
        xlabel = r'$u$ $(\mu m/s)$'
    elif x_var == 'time':
        x = samp_t
        xlabel = r'Time, $t$ $(s)$'
    elif x_var == 'strain':
        x = e
        xlabel = r'$\epsilon$'
    
    if y_var == 'shear':
        y = dudr
        ylabel = r'$\dot{\gamma}$ $(s^{-1})$'
    elif y_var == 'stress':
        y = stress
        ylabel = r'Applied Stress, $\sigma$ $(Pa)$'
    elif y_var == 'radius':
        y = samp_r*1000000
        ylabel = r'Deformed Radius, $R^{\mathbf{\prime}}$ $(\mu m)$'
    elif y_var == 'velocity':
        y = veloc*1000000
        ylabel = r'$u$ $(\mu m/s)$'
    elif y_var == 'time':
        y = samp_t
        ylabel = r'$t$ $(s)$'
    elif y_var == 'strain':
        y = e
        ylabel = r'Uniaxial Strain, $\epsilon$'
    
    
    ax.scatter(x, y, color='black', marker = 'o',s=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    
    pow_stress = pow_K * ((dudr)**pow_n)
    #plt.plot(dudr, pow_stress, color='blue', label='Power Law' f' ($\mathrm{{R}}^2$ = {round(pow_r2,3)})', linewidth=0.75)
    
    h_stress = (h_K * ((dudr)**h_n)) + h_y
    #plt.plot(dudr, h_stress, color='red', label='Herschel-Bulkley' f' ($\mathrm{{R}}^2$ = {round(h_r2,3)})', linewidth=0.75)
    
    c_stress = ((math.sqrt(c_y)) + np.sqrt(c_K*dudr))**2
    #plt.plot(dudr, c_stress, color='green', label='Casson' f' ($\mathrm{{R}}^2$ = {round(c_r2,3)})', linewidth=0.75)
    
    cp_stress = cassonPapanFunc(dudr, cp_mu, cp_y, cp_m)
    #plt.plot(dudr, cp_stress, color='orange', label='Casson-Papanastasiou' f' ($\mathrm{{R}}^2$ = {round(cp_r2,3)})', linewidth=0.75)
    
    if hp_r2 > 0.8:
        hp_stress = hershelPapanFunc(dudr, hp_n, hp_K, hp_m, hp_ys)
        plt.plot(dudr, hp_stress, color='magenta', label='HB-Papanastasiou ', linewidth=0.75)
    
    # Make the major tick points line up nicely
    if logScale:
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs='auto'))
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs='auto'))
    #else:      
        #x_ticks = ax.xaxis.get_majorticklocs()
        #y_ticks = ax.yaxis.get_majorticklocs()
    
        #ax.set_xlim(x_ticks[0], x_ticks[-1])
        #ax.set_ylim(y_ticks[0], y_ticks[-1])
        
    else:
        x_ticks = ax.xaxis.get_majorticklocs()
        y_ticks = ax.yaxis.get_majorticklocs()

        custom_xmin = 0  # <<< set your desired lower x-axis limit here
        custom_ymin = 800  # <<< set your desired lower y-axis limit here
        
        custom_xmax = 120  # <<< set your desired lower x-axis limit here
        custom_ymax = 1500  # <<< set your desired lower y-axis limit here

        ax.set_xlim(custom_xmin, custom_xmax)
        ax.set_ylim(custom_ymin, custom_ymax)
    
    plt.legend()
    plt.tight_layout()
    
    return
    
# %%% 10.2 Viscosity Ratio by Slips and Single Material
def viscCompareGraphsSlips(dataFrame, matCode):
    
    viscScale, error, R_shear, _ = viscCompareData(dataFrame, matCode)
    
    
    # !!! do MAKE THIS VARIABLE DEPENDING ON TOTAL AMOUNT OF SLIPS !!!
    #totalSlips = sorted(viscScale.keys()) 
    totalSlips = 3
    logScale = True
    fig, ax, = figSettings(logScale, R_shear, (viscScale[1], viscScale[2], viscScale[3]))
    
    for i in range(totalSlips):
        
        ax.scatter(R_shear, viscScale[i+1], color = Colours[i], label = str(i+1) + ' Slip(s)', s = 3, marker=Markers[i])
        plt.errorbar(R_shear, viscScale[i+1], yerr = error[i+1], ls='none', capsize=2, elinewidth=0.5, color = Colours[i])
        
    plt.xlabel(r'$\dot{\gamma}$ $(s^{-1})$')
    plt.ylabel(r'PR:R $\mu_{eff}$ Scale Factor')

    plt.legend()
    
    return
    
# %%% 10.3 Viscosity Ratio by Material, Averaged Over Slips    
def viscCompareGraphsAvgs(dataFrame, matCodes, dataOnly = False):
    

    avgScaleBounds = []
    logScale = True
    avgScale = {}
    stdScale = {}
    
    for matCode in matCodes:
        
        
        viscScale, error, R_shear, nRepeats = viscCompareData(dataFrame, matCode)
        
        
    
        length = len(R_shear)
        avgScale[matCode] = [0]*length
        stdScale[matCode] = [0]*length
        
        # Get available slip numbers by sorting the viscosity dictionary and pulling the names of the sub-dictionaries (which defined earlier are slip no.)
        # slip_num = sorted(viscScale.keys())
        
        for j in range(length):
            
            
            # Gather viscosity values from all slips that occur at the same position
            # comb = np.array([viscScale[slip][j] for slip in slip_num])
            # stdComb = np.array([error[slip][j] for slip in slip_num])
            
            # Find how many repeats are present in each slip
            # repeats = [nRepeats[slip] for slip in slip_num]
            
            avgScale[matCode][j] = viscScale[j]
            # stdScale[matCode][j] = avgStdDev(stdComb)
            # stdScale[matCode][j] = weighted_std(stdComb, repeats)
            # stdScale[matCode][j] = pooledStd(stdComb, repeats)
            stdScale[matCode][j] = error[j]
            
        avgScaleBounds.append(avgScale[matCode])

            
    if not dataOnly:
        fig, ax = figSettings(logScale, R_shear, avgScaleBounds)    
    bestFit = {}
    
    for ind, matCode in enumerate(matCodes):
        
        bestFit[matCode] = [0]*3
        
        
        bestFit[matCode] = logFit(R_shear, avgScale[matCode])
        
        if not dataOnly:
        
            pred_y = [0]*len(R_shear)
            pred_y = np.exp(bestFit[matCode][1])*(R_shear**bestFit[matCode][0])
            
            SF = np.exp(bestFit[matCode][1])
            coeff, exponent = f"{SF:.1e}".split("e")
            sci_SF = f'{coeff} × 10^{{{int(exponent)}}}'
            eq = f'PR:R = ${sci_SF}$ $\dot{{\gamma}}^{{{round(bestFit[matCode][0], 2)}}}$'
            Label = matSelector(matCode, True) + ':    ' + eq + '    (R$^2$ = '+str(round(bestFit[matCode][2],4)) + ')'
            
            
            # !!! To upgrade to use fill_between instead of error bars as they are so close, or overlay slip curves lightly to see in background
            ax.scatter(R_shear, avgScale[matCode], color = Colours[ind], marker = Markers[ind], s = 9, label = Label, facecolors = 'none', linewidths = 0.5)         
            plt.errorbar(R_shear, avgScale[matCode], yerr = stdScale[matCode], ls='none', capsize=2, elinewidth=0.5, color = Colours[ind])
            ax.plot(R_shear, pred_y, color = Colours[ind], lw = 1, ls = 'solid')
    
        
    if not dataOnly:
        
        plt.xlabel(r'$\dot{\gamma}$ $(s^{-1})$')
        plt.ylabel(r'PR:R $\mu_{eff}$ Scale Factor')
    
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend( prop={'size': 4.4})
    
    return bestFit, stdScale
    

# %%% 10.4 Yield Stress Box Plot by Material
def yieldStressBoxPlot(dataFrame, matCodes):
    
    
    ID = dataFrame.get('ID')
    h_ys = dataFrame.get('Hershel Yield')
    h_r2 = dataFrame.get('Hershel R2')
    c_ys = dataFrame.get('Casson Yield')
    c_r2 = dataFrame.get('Casson R2')
    cp_ys = dataFrame.get('Cas-Papan Yield')
    cp_r2 = dataFrame.get('Cas-Papan R2')
    
    yieldStresses = {}
    
    for matCode in matCodes:
        
        matMask = np.array([str(i)[0] == matCode for i in ID])
        name = matSelector(matCode, True)
        
        yieldStresses[name] = {}
        H_ys = h_ys[matMask]
        C_ys = c_ys[matMask]
        CP_ys = cp_ys[matMask]
        H_r2 = h_r2[matMask]
        C_r2 = c_r2[matMask] 
        CP_r2 = cp_r2[matMask]
        
        filt_indices = np.where(matMask)[0]
        
        for i in filt_indices:
            
            R2 = [0]*2
            R2 = np.array((H_r2[i], C_r2[i], CP_r2[i]))
            opt = np.argmax(R2)
            
            if opt == 0: # 0 HB, 1 Casson, 2 CP
                
                yieldStresses[name][i] = H_ys[i]
               
            elif opt == 1:
               
                yieldStresses[name][i] = C_ys[i]
                
            elif opt == 2:
                
                yieldStresses[name][i] = CP_ys[i]
               
        fig, ax = plt.subplots()
        
        

        ax = sns.boxplot(yieldStresses, width = 0.3, linewidth=1,  showmeans = True,
                    boxprops=dict(edgecolor='black', facecolor='white'),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    medianprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='x', markerfacecolor='None', markersize=3,  markeredgecolor='black'),
                    meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor = 'black', markersize=3))
        
        handles, labels = ax.get_legend_handles_labels()
        
        # Create a single custom "Mean" legend handle
        mean_handle = mlines.Line2D([], [], color='black', marker='o', markerfacecolor='white',
                        markeredgecolor='black', markersize=3, linestyle='None', label='Mean')

        # Append it once
        handles.append(mean_handle)
        labels.append('Mean Average')
        
        ax.legend(handles, labels)
        
        plt.ylabel('Yield Stress (Pa)')
        # plt.grid(False)
        # plt.gca().spines['top'].set_color('black')
        # plt.gca().spines['right'].set_color('black')
        # plt.gca().spines['bottom'].set_color('black')
        # plt.gca().spines['left'].set_color('black')
        ax.tick_params(axis='y', direction='in', length = 2, width=1)


        #MAKE IT AUTO FIND MAX AT LATER DATE!!!
        ax.set_ylim(ymin=0)

        
    return 


# %%% 10.5 R2 Violin/Box Plot by Model and Material
def modelFitViolin(dataFrame, matCodes, box = False, ymin=0.5):
    
    # # Filter out everything bar the R2 values
    # r2_keys = [key for key in model_keys if key.endswith('r2')]
    
    # # Find the R2 values within the dataframe, by only extracting the keys that have R2 at the end (.lower makes this r2)
    # r2_columns = [col for col in dataFrame.columns if any(key in col.lower().replace(' ', '_') for key in r2_keys)]
    
    keys = ['Power R2', 'Hershel R2', 'Casson R2', 'Cas-Papan R2', 'Her-Papan R2']
    
    R2data = modelData_extraction(dataFrame, keys)
    ID = dataFrame['ID']

    boxplot_data = []
    
    for model in keys:
        values = R2data[model]
        
        for mat_ID, val in enumerate(values):
            
            matCode = ID.iloc[mat_ID][0]
            
            if matCode in matCodes and val > 0.4:
                
                boxplot_data.append({
                    'Material' : matSelector(matCode, True),
                    'Model' : modelSelector(model.replace(' R2', ''), True), # Clean up the labels so it only shows the model label
                    'R2' : val})
                
    boxplot_dataFrame = p.DataFrame(boxplot_data)


    fig, ax = plt.subplots()
    
    if not box:
        sns.violinplot(boxplot_dataFrame, x = 'Material', y = 'R2', hue = 'Model', scale = 'width', palette=Colours,
                      fill=False, linewidth=1, inner_kws=dict(color='.4', box_width=2))
        
    else:
        ax = sns.boxplot(boxplot_dataFrame, x = 'Material', y = 'R2', hue = 'Model', palette=Colours,
                        fill=False, linewidth=1, showmeans= True, 
                        meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor = 'black', markersize=3),
                        flierprops=dict(marker='x', markerfacecolor='None', markersize=3,  markeredgecolor='black'))
        handles, labels = ax.get_legend_handles_labels()
        
        # Create a single custom "Mean" legend handle
        mean_handle = mlines.Line2D([], [], color='black', marker='o', markerfacecolor='white',
                        markeredgecolor='black', markersize=3, linestyle='None', label='Mean')

        # Append it once
        handles.append(mean_handle)
        labels.append('Mean Average')
    
        ax.legend(handles, labels)
    
    ax.set_ylim([ymin, 1])
    plt.tight_layout() 
    # plt.show()


    return matCode


# %%% 10.6 Rheometer Viscosity Graphs
def rheometerGraphs(matCodes, includeG = False):
    
    visc_mean = {}
    visc_std = {}
    shear_mean = {}
    shear_std = {}
    yvals = []
    xvals = []
    
    # G1 = {}
    # G2 = {}
    
    for mat in matCodes:
    
        material = matSelector(mat)
        
        visc_mean[material], visc_std[material], shear_mean[material], shear_std = rheometerData(mat)
        
        yvals.append(visc_mean[material])
        xvals.append(shear_mean[material])
        
        # if includeG:
            
        #     G1[material] = 

    fig, ax1 = figSettings(True, xvals, yvals)   
        
    
    for ind, mat in enumerate(matCodes):
        
        material = matSelector(mat)
        
        ax1.scatter(shear_mean[material], visc_mean[material], color=Colours[ind], marker=Markers[ind], s = 4, label = material)
        ax1.errorbar(shear_mean[material], visc_mean[material], yerr=visc_std[material], color=Colours[ind], ls='none', capsize=2, elinewidth=0.3, markeredgewidth=0.75)
        
        # if includeG:
            
            
        
    plt.legend()
    plt.xlabel(r'$\dot{\gamma}$ $(s^{-1})$')
    plt.ylabel(r'$\eta$ $(Pa\cdot s)$')
    
    return visc_mean
    

# %%% 10.7 Protorheology Viscosity Graphs
def protorheologyGraphs(dataFrame, matCodes, dataOnly = False, shearRange_base10 = None, increments = 20):
    
    # If desiring a custom shear range, it will be in base 10. 
    
    shear = {}
    visc_mean = {}
    visc_std = {}
    xvals = []
    yvals = []
    yerrs = []
    visc_mean2 = {}
    visc_std2 = {}
    nRepeats = {}
    All_avgVisc = {}
    All_stdVisc = {}
    nSize = {}
    
    # !!! Need to make it so the cell ones take the other shears, otherwise it don't work as there isnt any
    if shearRange_base10 is None:
        
        for mat in matCodes:
            
            material = matSelector(mat)
            
            shear[material], _ = rheometerData(mat, True)

            xvals.append(shear[material])
 
            
    elif shearRange_base10 is not None:
        
        for mat in matCodes:
            
            material = matSelector(mat)
            
            shear[material] = np.logspace(shearRange_base10[0], shearRange_base10[1], base=10, num=increments)

            xvals.extend(shear[material])    
    
    
    
    for mat in matCodes:
        
        material = matSelector(mat)
        
        visc_mean2[mat] = {}
        visc_std2[mat] = {}
        All_avgVisc[mat] = {}
        
        
        
        visc_mean[material], visc_std[material], nRepeats[mat], All_avgVisc[mat], All_stdVisc[mat], nSize[mat] = allViscosity(dataFrame, mat, shear[material])
            
        # Get available slip numbers by sorting the viscosity dictionary and pulling the names of the sub-dictionaries (which defined earlier are slip no.)
        slip_num = sorted(visc_mean[material].keys())
        
        length = len(shear[material])
        

        for j in range(length):
            #print('debug ' + str(j) + ' ID ' +dataFrame['ID'][j])
            # Gather viscosity values from all slips that occur at the same position
            comb = np.array([visc_mean[material][slip][j] for slip in slip_num])
            stdComb = np.array([visc_std[material][slip][j] for slip in slip_num])
            
            repeats = np.array([nRepeats[mat][slip] for slip in slip_num])
            
            visc_mean2[mat][j] = np.mean(comb)
            # visc_std2[mat][j] = avgStdDev(stdComb)
            # visc_std2[mat][j] = weighted_std(stdComb, repeats)
            visc_std2[mat][j] = pooledStd(stdComb, repeats)
        
        
        yvals.append(np.array(list(visc_mean2[mat].values())))
        yerrs.append(np.array(list(visc_std2[mat].values()))) 
    
    if dataOnly:
        
        return visc_mean2, visc_std2, shear, nRepeats
    
    fig, ax= figSettings(True, xvals, yvals)
    SEM = {}
    
    for ind, mat in enumerate(matCodes):
        
        material = matSelector(mat)
        
        SEM[mat] = All_stdVisc[mat]/np.sqrt(nSize[mat]) # 
        
        ax.scatter(shear[material], All_avgVisc[mat], color=Colours[ind], marker=Markers[ind], s = 4, label = material)
        plt.errorbar(shear[material], All_avgVisc[mat], yerr= SEM[mat], color=Colours[ind], ls='none', capsize=2, elinewidth=0.3, markeredgewidth=0.75)
        
    plt.legend()
    plt.xlabel(r'$\dot{\gamma}$ $(s^{-1})$')
    plt.ylabel(r'$\eta $ $(Pa\cdot s)$')
            
        
        
    return

# %%% 10.8 Young's Modulus Graphs (Agarose 2.5%)       

def YoungsModulusGraphs(matCode, bySlip = False, byCell = False, pick = 0):
    
    Data = {}
    
    # Read input file
    Data['Inputs'] = p.read_excel('InputDataECalc2.xlsx', sheet_name='Sheet2')
    
    inputKeys = ['ID', 'Folder', 'Filename', 'SlipNo.', 'Start Radius', 'End Radius']
    
    if byCell:
        inputKeys = inputKeys + ['Cell Composition']
    
    inputs = modelData_extraction(Data['Inputs'], inputKeys)
    
    matCodes = [str(code).strip() for code in matCode]
    mask = np.array([str(i).strip().startswith(tuple(matCodes)) for i in inputs['ID']])

    inputs = modelData_extraction(Data['Inputs'], inputKeys, mask)
    length = len(inputs['ID'])
    
    outKeys = ['ID', 'Strain', 'SlipNo.', 'Delta', 'Stress']
    outputs = {key: [0]*length for key in outKeys}
    outputs['ID'] = inputs['ID']
    outputs['SlipNo.'] = inputs['SlipNo.']
    if byCell:
        outputs['Cell Composition'] = inputs['Cell Composition']
    
    for i in range(length):
        
        outputs['Strain'][i], outputs['Delta'][i] = strainCalcStatic(inputs['Start Radius'][i], inputs['End Radius'][i])
        
        outputs['Stress'][i] = stressCalc(inputs['Start Radius'][i], inputs['SlipNo.'][i])
        

    if bySlip and not byCell:
        
        fig, ax = figSettings(False, outputs['Strain'] + [0] + [0], outputs['Stress'] + [0]+[8000])
        
        dataFrame = p.DataFrame(outputs)
        
        grouped = dataFrame.groupby('SlipNo.')
        mean_strain = []
        mean_stress = []
        std_strain = []
        std_stress = []
        
        for i, (slip, group) in enumerate(grouped):
            
            mean_strain.append(group['Strain'].mean())
            std_strain.append(group['Strain'].std())
            mean_stress.append(group['Stress'].mean())
            std_stress.append(group['Stress'].std())
            
            ax.scatter(group['Strain'].mean(), group['Stress'].mean(), color= Colours[i], marker = Markers[i], s=9, label = int(slip), linewidths = 0.75)
            ax.errorbar(group['Strain'].mean(), group['Stress'].mean(), yerr=group['Stress'].std(), xerr = group['Strain'].std(), ls='none', capsize=2, elinewidth=0.5, color = Colours[i])
                 
            
        mean_strain = np.array(mean_strain)
        mean_stress = np.array(mean_stress)
        std_strain = np.array(std_strain)
        std_stress = np.array(std_stress)
    
        m = linearFitThroughZero(mean_strain, mean_stress)
        predx = np.linspace(0, max(outputs['Strain']))
        predy = predx *m 
        predy_data = np.array(mean_strain)*m
        
        r2 = r2Calc(mean_stress, predy_data)
        
        # ax.scatter(mean_strain, mean_stress, color= Colours[0], marker = Markers[5], s=7, facecolor=None)
        ax.plot(predx, predy, color = Colours[0], ls = 'dashed', lw= 0.5) 
        
        # plt.errorbar(mean_strain, mean_stress, yerr=std_stress, xerr = std_strain, ls='none', capsize=2, elinewidth=0.5, color = Colours[0])
    
    elif byCell and not bySlip:
        
        Shades = [['blue', 'darkblue'], ['red', 'darkred']]
        
        fig, ax = figSettings(False, outputs['Strain'] + [0] + [0], outputs['Stress'] + [0])
        
        mats = [id_[0] for id_ in outputs['ID']]
        unique_mats = set(mats)
        
        outputs2 = {m: {key: [] for key in outputs if key != 'ID'} for m in unique_mats}
        
        for x, m, in enumerate(mats):
            for key in outputs:
                if key != 'ID':
                    outputs2[m][key].append(outputs[key][x])

        for i, mat in enumerate(matCode):
            
            dataFrame = p.DataFrame(outputs2[mat])
        
            grouped = dataFrame.groupby('Cell Composition')
        
            for j, (cell_comp, data) in enumerate(grouped):
                
                material = matSelector(mat, True)
                
                m = linearFitThroughZero(data['Strain'], data['Stress'])
                predx = np.linspace(0, max(data['Strain']))
                predy = predx *m 
                predy_data = np.array(data['Strain'])*m
                
                r2 = r2Calc(data['Stress'], predy_data)
                
                note = f'{material} {cell_comp}% E = {int(round(m,0))} Pa $R^2$ = {round(r2,3)}'
                
                plt.scatter(data['Strain'], data['Stress'], color = Shades[i][j], marker = Markers[i], 
                            s = 3, label = note)

                plt.plot(predx, predy, color = Shades[i][j], ls ='solid', lw= 0.5)
                
                
                # print(material + ': ' + str(int(cell_comp)) + ' %' + ' --> R2 = ' + str(round(r2,4)) +
                      # ' E = ' + str(int(round(m,0))))
                
                
    elif not bySlip and not byCell:
    
        fig, ax = figSettings(False, outputs['Strain'] + [0] + [0], outputs['Stress'] + [0]+[8000])    
    
        m = linearFitThroughZero(outputs['Strain'], outputs['Stress'])
        predx = np.linspace(0, max(outputs['Strain']))
        predy = predx *m 
        predy_data = np.array(outputs['Strain'])*m
            
        
        r2 = r2Calc(np.array(outputs['Stress']), predy_data)
        print(r2)
        
        ax.set_ylim([0, 6000])
        # ax.set_xlim([0, 0.3])
        
        if pick != 3:
            ax.scatter(outputs['Strain'], np.array(outputs['Stress']), color= Colours[0], marker = Markers[0], s=7, facecolor='none', lw=0.5)
            ax.plot(predx, predy, color = Colours[0], ls = 'dashed', lw= 0.5, label = 'Best fit') 
    
        
        maskA = np.array([str(i)[0] == 'A' for i in outputs['ID']])
        maska = np.array([str(i)[0] == 'a' for i in outputs['ID']])
        
        Ea = {}
        EA = {}
        
        E = np.array(outputs['Stress'])/np.array(outputs['Strain'])
        Ea = np.array(outputs['Stress'])[maska]/np.array(outputs['Strain'])[maska]
        EA = np.array(outputs['Stress'])[maskA]/np.array(outputs['Strain'])[maskA]
        
        E_mean = np.mean(E)
        Ea_mean = np.mean(Ea)
        EA_mean = np.mean(EA)
        
        E_std = np.std(E)
        Ea_std = np.std(Ea)
        EA_std = np.std(EA)
        
        strain = np.linspace(0, max(outputs['Strain']))
        
        
        
        if pick == 0:
            ax.plot(strain, ((E_mean+E_std)*strain), ls = 'dotted', color ='black', lw = 0.5, label = r'$\pm$ $\sigma$')
            ax.plot(strain, ((E_mean-E_std)*strain), ls = 'dotted', color ='black', lw = 0.5)
        elif pick == 1:
            ax.plot(strain, ((Ea_mean+Ea_std)*strain), ls = 'dotted', color ='black', lw = 0.5, label = r'$\pm$ $\sigma$')
            ax.plot(strain, ((Ea_mean-Ea_std)*strain), ls = 'dotted', color ='black', lw = 0.5)
        elif pick == 2:
            ax.plot(strain, ((EA_mean+EA_std)*strain), ls = 'dotted', color ='black', lw = 0.5, label = r'$\pm$ $\sigma$')
            ax.plot(strain, ((EA_mean-EA_std)*strain), ls = 'dotted', color ='black', lw = 0.5)
        
        elif pick == 3:
            
            
            straina = np.linspace(0, max(np.array(outputs['Strain'])[maska]))
            ma = linearFitThroughZero(np.array(outputs['Strain'])[maska], np.array(outputs['Stress'])[maska])
            predxa = np.linspace(0, max(np.array(outputs['Strain'])[maska]))
            predya = predxa *ma
            predya_data = (np.array(outputs['Strain'])[maska])*ma
            r2a = r2Calc(np.array(outputs['Stress'])[maska], predya_data)
        
            exponenta = int(np.floor(np.log10(ma)))
            coefficienta = ma / (10**exponenta)
            slope_texta = r'$\mathrm{{E}}_{{0.5}}$ = {:.1f} $\times$ 10$^{{{}}}$ Pa'.format(coefficienta, exponenta)
        
            ax.plot(straina, ((Ea_mean+Ea_std)*straina), ls = Linestyles[0], color =Colours[2], lw = 0.5)
            ax.plot(straina, ((Ea_mean-Ea_std)*straina), ls = Linestyles[0], color =Colours[2], lw = 0.5)
            ax.plot(predxa, predya, color = Colours[2], ls = 'solid', lw= 0.5, label = slope_texta + f', $\mathrm{{R}}^2$ = {round(r2a,3)}')
            ax.scatter(np.array(outputs['Strain'])[maska], np.array(outputs['Stress'])[maska], color= Colours[2], marker = Markers[1], s=7, lw=0.75, label = '0.5 $\mu$l')
            
            strainA = np.linspace(0, max(np.array(outputs['Strain'])[maskA]))
            mA = linearFitThroughZero(np.array(outputs['Strain'])[maskA], np.array(outputs['Stress'])[maskA])
            predxA = np.linspace(0, max(np.array(outputs['Strain'])[maskA]))
            predyA = predxA *mA
            predyA_data = (np.array(outputs['Strain'])[maskA])*mA
            r2A = r2Calc(np.array(outputs['Stress'])[maskA], predyA_data)
            
            exponentA = int(np.floor(np.log10(mA)))
            coefficientA = mA / (10**exponentA)
            slope_textA = r'$\mathrm{{E}}_{{2}}$ = {:.1f} $\times$ 10$^{{{}}}$ Pa'.format(coefficientA, exponentA)
            
            ax.plot(strainA, ((EA_mean+EA_std)*strainA), ls = Linestyles[0], color =Colours[1], lw = 0.5)
            ax.plot(strainA, ((EA_mean-EA_std)*strainA), ls = Linestyles[0], color =Colours[1], lw = 0.5)
            ax.plot(predxA, predyA, color = Colours[1], ls = 'solid', lw= 0.5, label = slope_textA + f', $\mathrm{{R}}^2$ = {round(r2A,3)}')
            ax.scatter(np.array(outputs['Strain'])[maskA], np.array(outputs['Stress'])[maskA], color= Colours[1], marker = Markers[0], s=8, facecolor='none', lw=0.75, label = '2 $\mu$l')
            
            
            # ax.plot(strain, ((E_mean+E_std)*strain)/1000, ls = 'dotted', color ='black', lw = 0.5, label = r' Grouped $\pm$ $\sigma$')
            # ax.plot(strain, ((E_mean-E_std)*strain)/1000, ls = 'dotted', color ='black', lw = 0.5)
            # ax.plot(predx, predy/1000, color = Colours[0], ls = 'dashed', lw= 0.5, label = 'Best fit')
        
        
        # ax.scatter(outputs['Strain'], outputs['Stress'], color= Colours[0], marker = Markers[5], s=3, facecolor='none')
        # ax.plot(predx, predy, color = Colours[0], ls = 'dashed', lw= 0.5) 

    leg = plt.legend(prop={'size': 5}, title = 'Number of Slips', fancybox=True, fontsize='small')
    leg._legend_box.align = 'left'
    plt.setp(leg.get_title(),fontsize='small')        
    
    
    plt.ylabel('$\sigma$ (Pa)')
    plt.xlabel('$\epsilon$')
   
    
    if not byCell:
        
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))  # Force scientific notation
        ax.yaxis.set_major_formatter(formatter)
        
        # t_predy = predx * 84000
        #ax.plot(predx, t_predy, color = Colours[0], ls = 'dotted', lw= 0.5, label = 'Ferraro et al.')
    
        
        exponent = int(np.floor(np.log10(m)))
        coefficient = m / (10**exponent)
        slope_text = r'E = {:.1f} $\times$ 10$^{{{}}}$ Pa'.format(coefficient, exponent)
        # t_exponent = 4
        # t_coeffecient = 8.4
        #theory_text = r'E (Ferraro et al.) = {:.1f} $\times$ 10$^{{{}}}$'.format(t_coeffecient, t_exponent)
        
        plt.annotate(f'R² = {r2:.3f}', xy=(0.05, 0.45), xycoords='axes fraction',
                 horizontalalignment='left', verticalalignment='top')
        plt.annotate(slope_text, xy=(0.05, 0.35), xycoords='axes fraction',
                 horizontalalignment='left', verticalalignment='top')
        #plt.annotate(theory_text, xy=(0.05, 0.8), xycoords='axes fraction',
        #         horizontalalignment='left', verticalalignment='top')
        
    return outputs['Strain'], outputs['Stress']


# %%% 10.9 Cell viscosity prediction 

def cellViscPredict(dataFrame, matCodes, cellCodes):
    
    bestFit = {}
    visc = {}
    visc_std = {}
    shear = {}
    cell = [0]*len(matCodes)
    SF= {}
    pred_visc = {}
    SF_std = {}
    pred_std = {}
    nRepeats_visc = {}
    nRepeats_SF = {}
    
    
    for mat in matCodes:
        
        bestFit[mat], SF_std[mat] = viscCompareGraphsAvgs(dataFrame, mat, True)
        _, _, _, nRepeats_SF[mat] = protorheologyGraphs(dataFrame, mat, True)
        
        
    for cell in cellCodes:
        
        cell_material = matSelector(cell)
        visc[cell], visc_std[cell], shear[cell], nRepeats_visc[cell] = protorheologyGraphs(dataFrame, cell, True)
        # visc[cell], visc_std[cell], shear[cell], nRepeats_visc[cell] = protorheologyGraphs(dataFrame, cell, True, shearRange_base10 = [-4, -3], increments = 20)
        
        if cell.lower() == 'o':
            
            SFcode = 'C'
            
        elif cell.lower() == 'i':
            
            SFcode = 'F'
        
        elif cell.lower() == 'y':
            
            SFcode = 'H'
            
        else:
            return 'Mismatch cell-mat error'
        
        
        SF[cell] = np.exp(bestFit[SFcode][SFcode][1])*(np.array(shear[cell][cell_material])**bestFit[SFcode][SFcode][0])
            
        pred_visc[cell] = np.divide(list(visc[cell][cell].values()), SF[cell])
        pred_std[cell] = [0] * len(pred_visc[cell])
        
        for j in range(len(pred_visc[cell])):
            
            # pred_std[cell][j] = (avgStdDev(np.array([SF_std[SFcode][SFcode][j], visc_std[cell][cell][j]]))/SF[cell][j])
            _, pred_std[cell][j] = errorPropogation(visc[cell][cell][j], SF[cell][j], visc_std[cell][cell][j], SF_std[SFcode][SFcode][j], 
                                                    np.sum(list(nRepeats_visc[cell][cell].values())), 1) # !!! Do I add np.sum(list(nRepeats_SF[SFcode][SFcode].values())) here? Is it realistic

        
    yvals = np.array([value for cell in pred_visc.values() for value in cell])
    xvals = np.array([value for cell1 in shear.values() for cell2 in cell1.values() for value in cell2])
        
        
    fig, ax = figSettings(True, xvals, yvals)
    
    for i, cell in enumerate(cellCodes):
        
        cell_material = matSelector(cell)
        
        new_bestFit = logFit(shear[cell][cell_material], pred_visc[cell])
        A = np.exp(new_bestFit[1])
        coeff, exponent = f"{A:.1e}".split("e")
        sci_A = f'{coeff} × 10^{{{int(exponent)}}}'
        eq = f'$\mu_{{eff}}$ = ${sci_A}$ $\dot{{\gamma}}^{{{round(new_bestFit[0], 3)}}}$'
        Label = matSelector(cell, True) + ':    ' + eq + '    (R$^2$ = '+str(round(new_bestFit[2],4)) + ')'
        
        predy = A*(shear[cell][cell_material]**new_bestFit[0])
        
        ax.scatter(shear[cell][cell_material], pred_visc[cell], color= Colours[i], s=6, marker = Markers[i], label = Label, facecolor='none', lw=0.5)
        ax.errorbar(shear[cell][cell_material], pred_visc[cell], yerr = pred_std[cell], ls='none', capsize=2, elinewidth=0.5, color = Colours[i])
        ax.plot(shear[cell][cell_material], predy, color=Colours[i], lw=0.75)
        
        
    plt.xlabel('$\dot{\gamma}$ ($s^{-1}$)')
    plt.ylabel('$\eta_{eff}$ (Pa$\cdot$s)')
    plt.legend()
        
    return
        
# %%% 10.10 Rheometer Moduli graphs
def Ggraphs(matCodes):

    G1 = {}
    G2 = {}
    Gstar = {}
    G1_std = {}
    G2_std = {}
    Gstar_std = {}
    shear = {}    
    Gavg = {}    

    for mat in matCodes:

        material = matSelector(mat)
        
        data = p.read_excel("RheometerMASTER.xlsx", sheet_name=material+'_G')
        s_data = p.read_excel("RheometerMASTER.xlsx", sheet_name=material+'_Strain')
        
        G1[mat] = np.array(data.get('Avg G1'))
        G2[mat] = np.array(data.get('Avg G2'))
        Gstar[mat] = np.array(data.get('Avg Gs'))
        shear[mat] = np.array(s_data.get('Avg'))
        G1[mat] = G1[mat][1:]
        G2[mat] = G2[mat][1:]
        Gstar[mat] = Gstar[mat][1:]
        shear[mat] = shear[mat][1:]
        
        G1_std[mat] = np.array(data.get('Stdev G1'))
        G2_std[mat] = np.array(data.get('Stdev G2'))
        Gstar_std[mat] = np.array(data.get('Stdev Gs'))
        G1_std[mat] = G1_std[mat][1:]
        G2_std[mat] = G2_std[mat][1:]
        Gstar_std[mat] = Gstar_std[mat][1:]
        
        
    yval1 = [value for cell in G1.values() for value in cell] 
    yval2 = [value for cell in G2.values() for value in cell]     
    yval3 = [value for cell in Gstar.values() for value in cell] 
    xval = [value for cell in shear.values() for value in cell]
    errval = [value for cell in G2_std.values() for value in cell]
    
    fig, ax = figSettings(True, xval, [yval1, yval2, yval3], errval)
    
    for i, mat in enumerate(matCodes):
        
        material = matSelector(mat)
        
        ax.scatter(shear[mat], G1[mat], color = Colours[i], marker = Markers[i], s=5, label = material+" G'")
        ax.errorbar(shear[mat], G1[mat], yerr = G1_std[mat], ls='none', capsize=2, elinewidth=0.5, color = Colours[i])
        
        ax.scatter(shear[mat], G2[mat], color = Colours[i], marker = Markers[i], s=5, facecolor='none', label = material+" G''", lw = 0.5)
        ax.errorbar(shear[mat], G2[mat], yerr = G2_std[mat], ls='none', capsize=2, elinewidth=0.5, color = Colours[i])
        
        # ax.scatter(shear[mat], Gstar[mat], color = Colours[i], marker = Markers[i], s=5)
        # ax.errorbar(shear[mat], Gstar[mat], yerr = Gstar_std[mat], ls='none', capsize=2, elinewidth=0.5, color = Colours[i])
    
        Gavg[mat] = np.mean(Gstar[mat])
        E = Youngs(Gavg[mat])
        exponent = int(np.floor(np.log10(E)))
        coefficient = E / (10**exponent)
        
        plt.annotate(rf'E$_{{{mat}}} = ${round(E, 1)} Pa', xy=(0.05, 0.95 - 0.075*i), xycoords='axes fraction',
                 horizontalalignment='left', verticalalignment='top')

    plt.ylabel('Modulus (Pa)')
    plt.xlabel('$\dot{\gamma}$ ($s^{-1}$)')
    plt.legend()
    
    
    return Gstar

# %% GLOBAL VARIABLES

# Model variables: change to add more rheological models to the code
model_keys = ['pow_n', 'pow_K', 'pow_r2','h_n', 'h_K', 'h_y', 'h_r2', 'c_K', 'c_y', 'c_r2', 'cp_mu', 'cp_y',
        'cp_m', 'cp_r2', 'hp_n', 'hp_K', 'hp_m', 'hp_ys', 'hp_r2']

Colours = ['black', 'blue', 'red', 'magenta', 'orange', 'green', 'cyan', 'indigo'] # !!! To change
Markers = ('o', '^', 's', 'v', '*', 'X', '<', '>')
Linestyles = ('dotted', 'dashed', 'dashdot', (0, (5,10)))

# %%% Debug? (print stuff into log?) ----------------------------------------------------------------------------------------------------------
debug = False


# %% RUN OPTIONS
    
# %%% FIGURE PICKING --------------------------------------------------------------------------------------------------------------------------

# **** For creating the protorheology graphs separately (testing)
# stdGraphMaker()

# **** Prerequiste for functions marked with +. Creates dataframe of fitted rheological parameters (hershel-buckley etc.)
# combined, dataFrame = bulkFunction()

# **** + Protorheology predicted effective viscosity graphs. WILL TAKE RHEOMETER SHEAR UNLESS base 10 range specified e.g = [-2, 2] = 0.001 --> 100. Also optionally specify increment number
# test = protorheologyGraphs(dataFrame, ['c', 'C', 'H'], False ,shearRange_base10 = [-5, 0], increments = 20)
# test = protorheologyGraphs(dataFrame, ['c', 'C', 'o', 'O'])

# **** + For comparing a singular materials viscosity to the rheometer data separated between slip number
# viscCompareGraphsSlips(dataFrame, 'F')

# **** + For comparing the different Protorheology:Rheometer ratios against defined materials 
test = viscCompareGraphsAvgs(dataFrame, ['o'])

# **** + Boxplot for yield stresses
# test = yieldStressBoxPlot(dataFrame, ['o', 'O', 'i', 'I', 'y', 'Y'])

# **** + Boxplot of R2 values for different models and different materials
# test = modelFitViolin(dataFrame, ['C', 'c', 'H', 'L', 'a'], True, ymin = 0.75)

# **** Create rheometer graphs of complex viscosity vs shear rate for selected materials
# test = rheometerGraphs(['c', 'C', 'H', 'L', 'a'])

# **** Agoarose protorheology validaion graphs. Put True if you want averages by slip
# test  = YoungsModulusGraphs(['a', 'A'], False, False, 3)
# test  = YoungsModulusGraphs(['o', 'O'], False, True)

# **** + To predict cell viscosities. First array is the scale factor material, second is the bioink
# test = cellViscPredict(dataFrame, ['C', 'H'], ['o', 'O', 'y', 'Y'])

# **** Rheometer modulus graphs
# test = Ggraphs(['C', 'H'])

# %%% MATCODES --------------------------------------------------------------------------------------------------------------------------
# (F) Fibrin = F
# (C) Collagen = C
# (FC) Fibrin blank = f
# (CC) Collagen blank = c
# (H) Hybrid = H
# (A-C) Alginate Calcium = L
# (A) Agarose 0.5% = a

# (FB-3) Cell 3 day Fibrin = i
# (CB-3) Cell 3 day Collagen = o
# (HB-3) Cell 3 day Hybrid = y
# (FB-7) Cell 7 day Fibrin = I
# (CB-7) Cell 7 day Collagen = O
# (HB-7) Cell 7 day Hybrid = Y



