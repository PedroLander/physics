import math
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

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

    def update(self, dt):
        # Update acceleration
        acc = [f / self.mass for f in self.total_forces]
        # Update speed
        self.speed = [v + (a * dt) for v, a in zip(self.speed, acc)]
        # Update position
        self.pos = [p + (v * dt) for p, v in zip(self.pos, self.speed)]

    
    def move(self):
        self.pos = [sum(c) for c in zip(self.pos, self.speed)]

b1 = Body("b1", [0, 0, 0], 5e24)       # Earth-like mass at origin
b2 = Body("b2", [1.5e7, 0, 0], 1e22)   # Smaller body at 15,000 km on x-axis
b3 = Body("b3", [0, 1.5e7, 0], 1e22)   # Smaller body at 15,000 km on y-axis

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

def update(frame):
    # Compute all forces
    temp_bodies = bodies.copy()
    for body in bodies:
        body.compute_force(temp_bodies)
    # Update all positions
    for body in bodies:
        body.update(dt=1000)
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

ani = FuncAnimation(fig, update, frames=500, interval=50, blit=False)
plt.show()

