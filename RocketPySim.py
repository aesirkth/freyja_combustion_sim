from rocketpy import Fluid, CylindricalTank, MassFlowRateBasedTank, HybridMotor
from rocketpy import Environment, Rocket, Flight



############################################ Hybrid

# fluids at 20°C
liquid_nox = Fluid(name="lNOx", density=786.6)
vapour_nox = Fluid(name="gNOx", density=159.4)

# Define tank geometry
tank_radius = 150 / 2000
tank_length = 0.75
tank_shape = CylindricalTank(tank_radius, tank_length)

# Define tank
burn_time = 7
nox_mass = 10
ullage_mass = nox_mass * 0.15
mass_flow = nox_mass / burn_time
print(mass_flow)
isp = 230
grain_length = 0.6
nozzle_length = 0.10
plumbing_length = 0.3

oxidizer_tank = MassFlowRateBasedTank(
    name="oxidizer tank",
    geometry=tank_shape,
    flux_time=burn_time - 0.01,
    initial_liquid_mass=nox_mass,
    initial_gas_mass=0,
    liquid_mass_flow_rate_in=0,
    liquid_mass_flow_rate_out= mass_flow,
    gas_mass_flow_rate_in=0,
    gas_mass_flow_rate_out=0,
    liquid=liquid_nox,
    gas=vapour_nox,
)

fafnir = HybridMotor(
    thrust_source = isp * 9.8 * mass_flow,
    dry_mass = 0, # accounted for in rocket weight
    dry_inertia = (0.125, 0.125, 0.002),
    nozzle_radius = 70 / 2000,
    grain_number = 1,
    grain_separation = 0,
    grain_outer_radius = 70 / 2000,
    grain_initial_inner_radius = 25 / 2000,
    grain_initial_height = 0.3,
    grain_density = 1.1,
    grains_center_of_mass_position = grain_length / 2 + nozzle_length,
    center_of_dry_mass_position = 0.284, # doesn't matter
    nozzle_position = 0,
    burn_time = burn_time,
    throat_radius = 26 / 2000, # why does this matter?
)

fafnir.add_tank(tank=oxidizer_tank, position=plumbing_length + grain_length + nozzle_length + tank_length / 2)




############################################ Flight

ground_level = 165
env = Environment(
    latitude=39.3897,
    longitude=-8.28896388889,
    elevation=ground_level,
    date=(2023, 10, 15, 12),
)

env.set_atmospheric_model("custom_atmosphere", wind_u=0, wind_v=-10)

freya = Rocket(
    radius=0.072,
    mass= 34, # Rocket (no casing)
    inertia=(4.6, 4.6, 0.015), # from open rocket
    power_off_drag=0.7, # from open rocket
    power_on_drag=0.7, # from open rocket
    center_of_mass_without_motor=1.43,
    coordinate_system_orientation="nose_to_tail"
)

freya.add_motor(fafnir, 3.10)

freya.add_nose(
    length=0.50,
    kind="Von Karman",
    position=0,
)

fins = freya.add_trapezoidal_fins(
    4,
    root_chord=0.2,
    tip_chord=0.1,
    span=0.1,
    position=2.7,
    sweep_angle=25
)

# freya.draw()

#freya.all_info()



#test_flight = Flight(rocket=freya, environment=env, rail_length=12, inclination=84, heading=0)
#fafnir.all_info()