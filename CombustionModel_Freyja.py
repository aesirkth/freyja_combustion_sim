import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#---------------------------------------------------------
# PARAMETERS AND CONSTANTS

g = 9.81
densityFuel = 1115
initialOxydizerMass = 10
Thrust = 2721
burnTime = 5
Isp = 172
O_F = 3.9

a=3.8e-5
n=0.5

#---------------------------------------------------------
# PORT GEOMETRY SPECIFICATIONS

# This model works for a circular port (hollow cylinder grain).
# In order to use the other geometries, we can obtain an "equivalent diameter"
# I.e., we take the port area of the geometry, and calculate the diameter that
# would yield that for a hollow cyllinder.

# Hollow cyllinder: A=706.86 mm^2 (D = 30 mm)
# Sinusoidal: A=714.68 mm^2 (D = 30.16 mm)
# Wagonwheel: A=525.97 mm^2 (D = 25.878 mm)

D_in_study = np.array([25.878,30,30.16]) * 1e-3
portLength = 0.262

#---------------------------------------------------------
# MASS FLOW CALCULATIONS

massFlow = Thrust / (Isp * g)
massFlowFuel = massFlow / (O_F + 1)
massFlowOx = massFlowFuel * O_F

#---------------------------------------------------------
# AUXILIARY FUNCTIONS

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

#---------------------------------------------------------
# MAIN FUNCTION

for D_in in D_in_study:
    t, r, _ = combustionSimulation(D_in)
    rate = np.gradient(r, t)
    plotResults(t, rate, 'Regression rate (mm/s)')
    print("A total of", round(np.trapz(rate,x=t)*1000),"mm of wall radius were consumed over the burntime")
