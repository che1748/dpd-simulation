import os
import numpy as np
import random
import matplotlib.pyplot as plt

# how do you pick cutoff radius?
# how do you find the direction of the force?

box = 30    # Define the box size
N = 50  # number of particles
rc = 1.0 # Cutiff radius
a = 35.0 # concervative force coefficient
gamma = 4.5 #Disspative force coefficient
sigma = 3.0 # Random force coefficient
KT = 1.0 # Thermal energy
dt = 0.01 # Time step

particles = np.random.rand(N,3) * box # An array that will store the locations of particles
velocities = np.random.randn(N,3) * 0.1 # small random velocities
forces = np.zeros((N,3))

# Create 50 particles randomly in the box and make sure they do not overlap
'''
for particle in particles:
    safety_limit = 0
    while particle in particles:
        particle = np.random.rand(3)*box
        safety_limit += 1

        if safety_limit > 1000:
            print("The particle cannot be put into the box without overlap")
            break
'''


def conservative_force(r, a, rc, r_ij):
    if 1e-10 < r < rc:  # avoid devision by 0
        r_hat = r_ij/r
        return a*(1.0 - r/rc)*r_hat
    return np.zeros(3)

def dissipative_force(r, gamma, rc, v_ij, r_ij):
    if 1e-10 < r < rc:
        r_hat = r_ij/r
        w_D = (1.0 - r/rc)**2
        return -gamma * w_D * r *np.dot(v_ij, r_hat) * r_hat
    return np.zeros(3)

def random_force(r, sigma, rc, dt, r_ij):
    if 1e-10 < r < rc:
        r_hat = r_ij/r
        w_R = 1.0 - r/rc
        theta = np.random.normal(0,1)
        return sigma * w_R * r * theta * r_hat / np.sqrt(dt)
    return np.zeros(3)

def minimum_image_distance(r_i, r_j, box_size):
    '''
    Apply periodic boundary conditions using minimum image convention
    '''
    dr = r_i - r_j
    dr = dr - box_size * np.round(dr/box_size)
    return dr

def calculate_forces():
    '''
    calculate all forces between particles pairs
    '''
    global forces
    for i in range(N):
        for j in range(i+1, N):
            r_ij = minimum_image_distance(particles[i], particles[j], box)
            r = np.linalg.norm(r_ij)

            if 1e-10 < r < rc:
                v_ij = velocities[i] - velocities[j]

                # Conservative force
                F_c = conservative_force(r,a,rc,r_ij)
                forces[i] += F_c 
                forces[j] -= F_c 

                # Dissipative force
                F_d = dissipative_force(r, gamma, rc, v_ij, r_ij)
                forces[i] += F_d
                forces[j] -= F_d

                # Random force
                F_r = random_force(r, sigma, rc, dt, r_ij)
                forces[i] += F_r
                forces[j] -= F_r
        return forces

def apply_periodic_boundaries():
    '''
    Apply periodic boundary conditions
    '''
    global particles
    particles %= box

def velocity_verlet():
    global particles, velocities, forces

    particles += velocities*dt + 0.5*forces*dt**2

    apply_periodic_boundaries()

    forces_new = calculate_forces()

    velocities += 0.5 * (forces + forces_new) *dt
    forces = forces_new




# Initialize the forces
calculate_forces()


def write_xyz_animation(particles_history, velocities_history, box_size, filename="animation.xyz"):
    """
    Write all frames to a single XYZ file for OVITO animation
    """
    with open(filename, 'w') as f:
        for step, (particles, velocities) in enumerate(zip(particles_history, velocities_history)):
            N = len(particles)
            
            # Write frame header
            f.write(f"{N}\n")
            f.write(f'Lattice="{box} 0 0 0 {box} 0 0 0 {box}" ')
            f.write(f'Properties=pos:R:3:vel:R:3 Step={step}\n')
            
            # Write particle data
            for i in range(N):
                pos = particles[i]
                vel = velocities[i]
                f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} ")
                f.write(f"{vel[0]:.6f} {vel[1]:.6f} {vel[2]:.6f}\n")

# Usage: Collect data during simulation, then write animation
def run_simulation_with_animation():
    particles_history = []
    velocities_history = []
    
    steps = 1000
    save_interval = 10  # Save more frequently for smooth animation
    
    for step in range(steps):
        velocity_verlet()
        
        # Collect data for animation
        if step % save_interval == 0:
            particles_history.append(particles.copy())
            velocities_history.append(velocities.copy())
        
        if step % 100 == 0:
            kinetic_energy = 0.5 * np.sum(velocities**2)
            temperature = 2.0 * kinetic_energy / (3.0 * N)
            print(f"Step {step}: Temperature = {temperature:.3f}")
    
    # Write animation file
    write_xyz_animation(particles_history, velocities_history, box)
    print(f"Animation saved with {len(particles_history)} frames")

# Run the simulation
run_simulation_with_animation()