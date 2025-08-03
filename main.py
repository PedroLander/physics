import math
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

G = 6.674E-11 # Gravitational Constant. (m³/(kg·s²))

class Body():
    def __init__(self, name, pos, mass):
        self.name = name
        self.pos = list(pos)
        self.mass = mass
        self.speed = [0, 0, 0]
        self.total_forces = [0, 0, 0]

        # History of positions for trail
        self.trail_x = []
        self.trail_y = []
        self.trail_z = []

    def compute_force(self, bodies):
        total_force = [0.0, 0.0, 0.0]
        for body in bodies:
            # Exclude self force
            if body is self:
                continue
            # Vector from body to self
            r_vec = [b - a for a, b in zip (self.pos, body.pos)]
            # Euclidean distance
            distance = math.sqrt(sum(c**2 for c in r_vec))
            # Exclude if in the same place (force infinite or zero)
            if distance == 0:
                continue
            # Force module
            force_mod = G * body.mass * self.mass / (distance**2)
            # Froce vector
            force_vec = [force_mod * (c / distance) for c in r_vec]
            # Update total force vector
            total_force = [f + df for f, df in zip(total_force, force_vec)]
        self.total_forces = total_force

    def euler(self, dt):
        acc = [f / self.mass for f in self.total_forces]
        # Update speed
        self.speed = [v + (a * dt) for v, a in zip(self.speed, acc)]
        # Update position
        self.pos = [p + (v * dt) for p, v in zip(self.pos, self.speed)]

    def runge_kutta2(self, dt, bodies):
        # Save current state
        orig_pos = self.pos[:]
        orig_speed = self.speed[:]
        acc1 = [f / self.mass for f in self.total_forces]
        # Estimate mid-point
        mid_speed = [v + 0.5 * a * dt for v, a in zip(orig_speed, acc1)]
        mid_pos = [p + 0.5 * v * dt for p, v in zip(orig_pos, orig_speed)]
        # Compute force at mid-point
        # Temporarily update state
        self.pos = mid_pos
        self.speed = mid_speed
        self.compute_force(bodies)
        acc2 = [f / self.mass for f in self.total_forces]
        # Restore state
        self.pos = orig_pos
        self.speed = orig_speed
        # Update using mid-point acceleration
        self.speed = [v + acc2[i] * dt for i, v in enumerate(orig_speed)]
        self.pos = [p + mid_speed[i] * dt for i, p in enumerate(orig_pos)]

    def runge_kutta4(self, dt, bodies):
        orig_pos = self.pos[:]
        orig_speed = self.speed[:]
        acc1 = [f / self.mass for f in self.total_forces]

        # k1
        k1_v = [a * dt for a in acc1]
        k1_x = [v * dt for v in orig_speed]

        # k2
        temp_speed = [orig_speed[i] + 0.5 * k1_v[i] for i in range(len(self.pos))]
        temp_pos = [orig_pos[i] + 0.5 * k1_x[i] for i in range(len(self.pos))]
        self.pos = temp_pos
        self.speed = temp_speed
        self.compute_force(bodies)
        acc2 = [f / self.mass for f in self.total_forces]
        k2_v = [a * dt for a in acc2]
        k2_x = [v * dt for v in temp_speed]

        # k3
        temp_speed = [orig_speed[i] + 0.5 * k2_v[i] for i in range(len(self.pos))]
        temp_pos = [orig_pos[i] + 0.5 * k2_x[i] for i in range(len(self.pos))]
        self.pos = temp_pos
        self.speed = temp_speed
        self.compute_force(bodies)
        acc3 = [f / self.mass for f in self.total_forces]
        k3_v = [a * dt for a in acc3]
        k3_x = [v * dt for v in temp_speed]

        # k4
        temp_speed = [orig_speed[i] + k3_v[i] for i in range(len(self.pos))]
        temp_pos = [orig_pos[i] + k3_x[i] for i in range(len(self.pos))]
        self.pos = temp_pos
        self.speed = temp_speed
        self.compute_force(bodies)
        acc4 = [f / self.mass for f in self.total_forces]
        k4_v = [a * dt for a in acc4]
        k4_x = [v * dt for v in temp_speed]

        # Restore state
        self.pos = orig_pos
        self.speed = orig_speed

        # Combine increments
        self.speed = [orig_speed[i] + (k1_v[i] + 2*k2_v[i] + 2*k3_v[i] + k4_v[i]) / 6 for i in range(len(self.pos))]
        self.pos = [orig_pos[i] + (k1_x[i] + 2*k2_x[i] + 2*k3_x[i] + k4_x[i]) / 6 for i in range(len(self.pos))]

    def update(self, dt, method="rk4", bodies=None):
        if method == "euler":
            self.euler(dt)
        elif method == "rk2":
            if bodies is None:
                raise ValueError("bodies must be provided for Runge-Kutta methods")
            self.runge_kutta2(dt, bodies)
        elif method == "rk4":
            if bodies is None:
                raise ValueError("bodies must be provided for Runge-Kutta methods")
            self.runge_kutta4(dt, bodies)
        else:
            raise ValueError(f"Unknown integration method: {method}")

    
    def move(self):
        self.pos = [sum(c) for c in zip(self.pos, self.speed)]

b1 = Body("b1", [0, 0, 0], 5e24)       # Earth-like mass at origin
b2 = Body("b2", [1.5e7, 0, 0], 1e23)   # Smaller body at 15,000 km on x-axis
b3 = Body("b3", [0, 1.5e7, 0], 1e23)   # Smaller body at 15,000 km on y-axis

bodies = [b1, b2, b3]

# Initial velocities roughly set for orbiting:
# For circular orbit velocity: v = sqrt(G * M / r)
def orbit_velocity(M, r):
    return math.sqrt(G * M / r)

# b2 orbiting b1 on y-axis velocity
b2.speed = [0, orbit_velocity(b1.mass, 1.5e7), 0]
# b3 orbiting b1 on -x axis velocity
b3.speed = [-orbit_velocity(b1.mass, 1.5e7), 0, 0]
# b1 initially stationary

# Plot setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatters = [ax.plot([], [], [], 'o', label=body.name)[0] for body in bodies]
trails = [ax.plot([], [], [], '-', linewidth=1)[0] for _ in bodies]
ax.set_xlim(-2e7, 2e7)
ax.set_ylim(-2e7, 2e7)
ax.set_zlim(-2e7, 2e7)
ax.legend()

def update(frame, bodies):
    # Compute all forces
    temp_bodies = bodies.copy()
    for body in bodies:
        body.compute_force(temp_bodies)
    # Update all positions
    for body in bodies:
        body.update(dt=300, bodies=bodies)
        # Add current position to trail
        body.trail_x.append(body.pos[0])
        body.trail_y.append(body.pos[1])
        body.trail_z.append(body.pos[2])

        # Limit trail length (optional)
        max_length = 300
        if len(body.trail_x) > max_length:
            body.trail_x.pop(0)
            body.trail_y.pop(0)
            body.trail_z.pop(0)

    # Update plot
    for i, body in enumerate(bodies):
        # Update scatter plot
        scatters[i].set_data([body.pos[0]], [body.pos[1]])
        scatters[i].set_3d_properties([body.pos[2]])

        # Update trail lines
        trails[i].set_data(body.trail_x, body.trail_y)
        trails[i].set_3d_properties(body.trail_z)

    return scatters + trails

ani = FuncAnimation(fig, partial(update, bodies=bodies), frames=500, interval=50, blit=False)
plt.show()

