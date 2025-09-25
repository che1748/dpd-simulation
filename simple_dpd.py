import numpy as np
import random
import matplotlib.pyplot as plt

# how do you pick cutoff radius?
# how do you find the direction of the force?

box = 100    # Define the box size
N = 50  # number of particles
rc = 1.0 # Cutiff radius
a = 35.0 # concervative force coefficient
gamma = 4.5 #Disspative force coefficient
sigma = 3.0 # Random force coefficient
KT = 1.0 # Thermal energy
dt = 0.01 # Time step

particles = np.zeros((N, 3)) # An array that will store the locations of particles
velocities = np.random.randn(N,3) * box # small random velocities
forces = np.zeros((N,3))

# Create 50 particles randomly in the box and make sure they do not overlap
for particle in particles:
    safety_limit = 0
    while particle in particles:
        particle = np.random.rand(3)
        safety_limit += 1

        if safety_limit > 1000:
            print("The particle cannot be put into the box without overlap")
            break



def conservative_force(r, a, rc):
    if 1e-10 < r < rc:  # avoid devision by 0
        return a*(1.0 - r/rc)
    return 0

def dissipative_force(r, gamma, rc, v_ij, r_ij):
    if 1e-10 < r < rc:
        r_hat = r_ij/r
        w_D = (1.0 - r/rc)**2
        return -gamma * w_D * np.dot(v_ij, r_hat) * r_hat
    return np.zeros(3)

def random_force(r, sigma, rc, dt, r_ij):
    if 1e-10 < r < rc:
        r_hat = r_ij/r
        w_R = 1.0 - r/rc
        theta = np.random.normal(0,1)
        return sigma * w_R * theta * r_hat / np.sqrt(dt)
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
    force = np.zeros((N,3))

    for i in range(N):
        for j in range(i+1, N):
            r_ij = minimum_image_distance(particles[i], particles[j], box)
            r = np.linalg.norm(r_ij)

            if 1e-10 < r < rc:
                v_ij = velocities[i] - velocities[j]

                # Conservative force
                F_c = conservative_force(r,a,rc)
                force_dir = r_ij/r
                forces[i] += F_c * force_dir
                forces[j] -= F_c * force_dir

                # Dissipative force
                F_d = dissipative_force(r, gamma, rc, v_ij, r_ij)
                force[i] += F_d
                force[j] -= F_d

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
forces = calculate_forces()

steps = 1000
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for step in range(steps):
    if step % 50 ==0:
        ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], c='blue', s=50)
        plt.savefig(f'sim{step}.pdf')
        ax.clear()
    velocity_verlet()

    if step % 100 == 0:
        # Calculate and print some statistics
        kinetic_energy = 0.5 * np.sum(velocities**2)
        temperature = 2.0 * kinetic_energy / (3.0 * N)
        print(f"Step {step}: Temperature = {temperature:.3f}")

# Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], c='blue', s=50)
ax.set_xlim(0, box)
ax.set_ylim(0, box)
ax.set_zlim(0, box)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('DPD Simulation - Final Configuration')
plt.show()
