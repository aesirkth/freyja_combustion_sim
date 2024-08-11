import math
import scipy
import os
import numpy as np
import RocketPySim
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from CEA_Wrap import Oxidizer, Fuel, RocketProblem
from CoolProp.CoolProp import PropsSI as props



def CEA(Temp_prop,Pressure,eps,O_F):
    n2o = Oxidizer("N2O", temp=Temp_prop,wt_percent=100) 
    c8h8 = Fuel("C8H8,styrene", temp=Temp_prop, wt_percent=50)
    c4h6 = Fuel("C4H6,butadiene",temp=Temp_prop,wt_percent=15)
    c3h3n1 = Fuel("C3H3N1",temp=Temp_prop,wt_percent=35,chemical_composition="C 3 H 3 N 1", hf=126.4) # Heat of formation in [kj/mol] is found here https://webbook.nist.gov/cgi/cbook.cgi?ID=C107131&Units=SI&Mask=E

    problem = RocketProblem(pressure=Pressure, materials=[n2o, c8h8,c4h6,c3h3n1], sup=eps, analysis_type="frozen nfz=2",o_f = O_F, pressure_units = "bar")
    results = problem.run()
    Pe = results.p
    
    Tc = results.c_t
    Pc = results.c_p
    Mm = results.c_mw  
    problem.set_analysis_type("frozen", nfz=1)
    results_frozen_cc = problem.run()
    cp = results_frozen_cc.c_cp
    
    R = 8314.462618
    gamma= cp/(cp- 8.314462618/Mm) #calculation of Gamma (aka Specific heat ratio)
    ve = math.sqrt((2*gamma)/(gamma - 1) *  (R/Mm)  * Tc*(1-(Pe/Pc)**((gamma-1)/gamma)))
    return Pe, ve, Tc
def regressionRate(t, v):
    r, oxydizerMass = v
    A = np.pi * r**2
    G_Ox = massFlowOx / A
    drdt = a * G_Ox**n
    return [drdt, massFlowOx]
def combustionSimulation(D_in):
    v0 = [D_in / 2, initialOxydizerMass]
    interval = (0, burnTime)
    sol = solve_ivp(regressionRate, interval, v0, method='RK45', t_eval=np.linspace(0, burnTime, 50))
    return sol.t, sol.y[0], sol.y[1]
def plotResults(t, rate, ylab):
    plt.figure()
    plt.title('Regression Rate over burntime')
    plt.plot(t, rate * 1000)
    plt.xlabel('Time (s)')
    plt.ylabel(ylab)
    plt.grid(True)
    plt.show()


############################## INITIAL DATA ###################################

#   ROCKET PY INITIAL DATA (CHANGE IT IN THE ROCKET PY SIM IF YOU WANT DIFFERENT THRUST ESTIMATION)
# Define tank geometry
#tank_radius = 150 / 2000
#tank_length = 0.7

# Define tank
#burn_time = 7
#nox_mass = 10
#ullage_mass = nox_mass * 0.15
#mass_flow = nox_mass / burn_time
#isp = 180
#grain_length = 0.6
#nozzle_length = 0.10
#plumbing_length = 0.3

T = RocketPySim.fafnir.average_thrust   #Average thrust estimation from rocketpy [N]
Cd = 0.7                                # Estimation for initial guess
D = 1.5*pow(10,-3)                      # Diameter of a single orifice [mm] 
L = 15                                  # Length of the injector [mm]
ch_param = 0.75                         # Choking parameter, used in the residence time formula (Need to verify)
A = pow(D,2)/4 * math.pi                # Area of a single injector orifice [mm2]
T_prop = 293                            # Temperature of the propellant and fuel (For CEA) [K]
DeltaP_Tank= 5*pow(10,5)                # Pressure drop between tank and injector -> Assumed could be calculated as particular energy losses at corners and lenght energy losses
DeltaP_Injector= 20*pow(10,5)           # Pressure drop between injector and combustion chamber
P1= 45*pow(10,5)                        # Pressure at the injector [Pa] Taken at 293 K After 5 Bar assumed drop in the line
P2 = P1-DeltaP_Injector                 # Pressure in the chamber  [Pa]
P_atm = 101325                          # Atmospheric pressure [Pa]
g = 9.81                                # Gravitational constant 
density_fuel = 1115                     # Density of the ABS used [kg/m3]
#D_t = 11.509375 * 10**-3                 # Throat diameter no.76 #29 nozzle [m2
#D_e = 31.877 * 10**-3                  # Exit diameter no.76 #40 nozzle [m2]

D_t = 15.875 * 10**-3                   # Throat diameter no.76 #40 nozzle [m2]
D_e = 38.1 * 10**-3                     # Exit diameter no.76 #40 nozzle [m2]



A_t = math.pi*(D_t/2)**2                # Throat area of no.76 #40 nozzle [m2]
print(A_t)
A_e = math.pi*(D_e/2)**2                # Exit area of no.76 #40 nozzle [m2]
print(A_e)


eps = A_e/A_t                           # Supersonic expansion ratio in the nozzle

print(eps)
OF_init = 3                             # Initial Oxidizer to fuel ratio
N = 0                                   # Initial number of the injector holes
a=1.6567e-4                             # Ballisic coefficient (Mario Amaro calculation from https://doi.org/10.2514/6.2014-3751, fig.15 )
n=0.56478                               # Ballisic coefficient (Mario Amaro calculation from https://doi.org/10.2514/6.2014-3751, fig.15 )
grain_length = 0.28                     # Length of the fuel grain [m]
initialOxydizerMass = 10                # 
burnTime = 7                            # Burn time of the rocket [s]


# PORT GEOMETRY SPECIFICATIONS

# This model works for a circular port (hollow cylinder grain).
# In order to use the other geometries, we can obtain an "equivalent diameter"
# I.e., we take the port area of the geometry, and calculate the diameter that
# would yield that for a hollow cyllinder.

# Hollow cyllinder: A=706.86 mm^2 (D = 30 mm)
# Sinusoidal: A=714.68 mm^2 (D = 30.16 mm)
# Wagonwheel: A=525.97 mm^2 (D = 25.878 mm)
D_in_study = np.array([25.878,30,30.16]) * 1e-3 
D = D_in_study[2]
A_port = ((D**2)/4) * math.pi



############################## ROUTINE START ###################################
[P_e,v_e,T_c] = CEA(T_prop,P2/pow(10,5),eps,OF_init)
m_dot_target = T/v_e - (P_e - P_atm)*A_e/(v_e)

m_dot_target = T/v_e - (P_e - P_atm)*A_e/(v_e)
mass_fu_target = m_dot_target / (OF_init + 1)
mass_ox_target = mass_fu_target* OF_init


m_dyer_dot = 0
while m_dyer_dot <= mass_ox_target:
    N = N+1
    #   DYER MODEL CALCULATION 

    h1 = props('H','P',P1,'Q',0,'N2O') # Saturated liquid enthalpy
    h2 = props('H','P',P2,'Q',0,'N2O') # Saturated liquid enthalpy
    rho1 = props('D','T',T_prop,'P',P1,'N2O')
    rho2 = props('D','T',T_prop,'P',P2,'N2O')
    P_v = props('P','T',T_prop,'Q',1,'N2O') # Saturated vapor pressure
    
    m_hem_dot = N*Cd*A*rho2*math.sqrt(2*(h1-h2)) # Equation for HEM mass flow rate, (2.50) from Waxman Doctoral thesis
    m_spi_dot = N*Cd*A*math.sqrt(2*rho1*(P1-P2)) #Equation for SPI mass flow rate, (2.17) from Waxman Doctoral thesis
    k = math.sqrt((P1-P2)/(P_v-P2))
    m_dyer_dot = ((k/(1+k)) * m_spi_dot + (1/(1+k))*m_hem_dot)
print("Oxidizer Mass flow rate with", N,"number of holes =", m_dyer_dot)


G_Ox = m_dyer_dot/A_port
r_dot = a * G_Ox**n
m_fuel_dot = 2*math.pi*grain_length*D/2 * density_fuel * r_dot

# Regression rate calculation (fuel mass flow rate calculation)
while m_fuel_dot <= mass_fu_target or G_Ox >= 700:  # Condition to find the needed mass flow rate and to not exceed the mass flux limit 
    if G_Ox <= 700:
        grain_length = grain_length + 0.001
    else:
        A_port = A_port+0.00000001                      # Iteration of port area to accomodate the mass flux 
    
    D = math.sqrt(A_port/math.pi * 4)               # Recalculation of diameter with new area
    G_Ox = m_dyer_dot/A_port                        # Calculation of mass flux
    r_dot = a * G_Ox**n                             # Burn rate estimation
    m_fuel_dot = math.pi*grain_length*D * density_fuel * r_dot      # Mass flow rate calculation
    
    
# Check for the respect of the initial O/F ratio
if(m_dyer_dot/m_fuel_dot != OF_init):     # If statement for checking if the O/F ratio is respected. If not it will adjust the grain length to the one that will respect it
     grain_length = (m_dyer_dot/OF_init)/(math.pi*D * density_fuel * r_dot) 
m_fuel_dot = math.pi*grain_length*D * density_fuel * r_dot  # Recalculation of the fuel mass flow rate with a new grain length
    
    

print("Fuel Mass flow rate with", N,"number of holes =", m_fuel_dot, "[kg/s] \n")
print("Area needed for the proper mass flux", A_port*10**6,"[mm^2] \n")
print("Diameter needed for the proper mass flux", D*1000,"[mm] \n")
print("Length of the fuel grain neeeded", grain_length*100,"[cm] \n")
print("Mass flux",G_Ox, "[kg/(s*m^2)] \n")
print("OF ratio", m_dyer_dot/m_fuel_dot)

massFlow = m_dyer_dot+m_fuel_dot
massFlowFuel = m_fuel_dot
massFlowOx = m_dyer_dot

Isp = T/massFlow/g
print("Isp:", Isp)

# Сombustion calc
t, r, _ = combustionSimulation(D)
rate = np.gradient(r, t)
print("A total of", round(np.trapz(rate,x=t)*1000),"mm of wall radius were consumed over the burntime")

def ChangeTempFixInjector(Temp_):
    #############################
    # Compares the initial calculated values: m_dyer_dot, m_fuel_dot, G_Ox, OF ratio and Thrust,
    # with the values obtained for the same injector geometry with a different temperature
    # Input: Temperature [K]
    #############################

    print("###########################################################")
    print("Initial Values. T_initial:", T_prop)    
    print("Oxidizer Mass flow rate with", N,"number of holes =", m_dyer_dot)
    print("Fuel Mass flow rate with", N,"number of holes =", m_fuel_dot, "[kg/s]")
    print("Mass flux",G_Ox, "[kg/(s*m^2)]") 
    print("OF ratio", m_dyer_dot/m_fuel_dot)
    print("A total of", round(np.trapz(rate,x=t)*1000),"mm of wall radius were consumed over the burntime")
    print("Initial thrust", T)
    print("###########################################################")

    P1_= 45*pow(10,5)
    P2_= P1_-DeltaP_Injector

    #   DYER MODEL CALCULATION 

    h1_ = props('H','P',P1_,'Q',0,'N2O') # Saturated liquid enthalpy
    h2_ = props('H','P',P2_,'Q',0,'N2O') # Saturated liquid enthalpy
    rho1_ = props('D','T',Temp_,'P',P1_,'N2O')
    rho2_ = props('D','T',Temp_,'P',P2_,'N2O')
    P_v_ = props('P','T',Temp_,'Q',1,'N2O') # Saturated vapor pressure
    
    m_hem_dot_ = N*Cd*A*rho2_*math.sqrt(2*(h1_-h2_)) # Equation for HEM mass flow rate, (2.50) from Waxman Doctoral thesis
    m_spi_dot_ = N*Cd*A*math.sqrt(2*rho1_*(P1_-P2_)) #Equation for SPI mass flow rate, (2.17) from Waxman Doctoral thesis
    k_ = math.sqrt((P1_-P2_)/(P_v_-P2_))
    m_dyer_dot_ = ((k_/(1+k_)) * m_spi_dot_ + (1/(1+k_))*m_hem_dot_)

    print("Oxidizer Mass flow rate with", N," number of holes and Temp:",Temp_,"ºC =", m_dyer_dot_)
    
    G_Ox_ = m_dyer_dot_/A_port
    r_dot_ = a * G_Ox_**n    #Balistic coeff * (Oxidizer/unitArea)^2ndBalistic electronic
    m_fuel_dot_ = 2*math.pi*grain_length*D/2 * density_fuel * r_dot_
    print("Fuel Mass flow rate with", N,"number of holes =", m_fuel_dot_, "[kg/s]")
    print("Mass flux",G_Ox_, "[kg/(s*m^2)]")
    print("OF ratio", m_dyer_dot_/m_fuel_dot_)

    
    [P_e_,v_e_,T_c_] = CEA(Temp_,P2_/pow(10,5),eps,m_dyer_dot_/m_fuel_dot_)

    t_, r_, _ = combustionSimulation(D)
    rate_ = np.gradient(r_, t_)
    
    print("A total of", round(np.trapz(rate_,x=t_)*1000),"mm of wall radius were consumed over the burntime")
    m_dot_target = T/v_e_ - (P_e - P_atm)*A_e/(v_e_)
    m_dot_tot_ = m_fuel_dot_ + m_dyer_dot_
    T_ = v_e_*m_dot_tot_ + (P_e_ - P_atm)*A_e
    print("Modified thrust", T_)
    print("###########################################################")
    #plotResults(t_, rate_, 'Regression rate (mm/s)')
    return

ChangeTempFixInjector(298)

#---------------------------------------------------------

plotResults(t, rate, 'Regression rate (mm/s)')