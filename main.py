import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------
# Utility / solver helpers
# -----------------------
def idx(i, j, N):
    """Return index with padded boundaries (not used here; kept for clarity)."""
    return i, j

def set_bnd(b, x, N):
    """
    Set boundary conditions.
    b == 0: scalar (density) -> Neumann (copy)
    b == 1: horizontal velocity -> flip at left/right
    b == 2: vertical velocity -> flip at top/bottom
    """
    # Left/right
    x[0, 1:-1] = -x[1, 1:-1] if b == 1 else x[1, 1:-1]
    x[-1, 1:-1] = -x[-2, 1:-1] if b == 1 else x[-2, 1:-1]
    # Top/bottom
    x[1:-1, 0] = -x[1:-1, 1] if b == 2 else x[1:-1, 1]
    x[1:-1, -1] = -x[1:-1, -2] if b == 2 else x[1:-1, -2]

    # Corners - average of neighbors
    x[0, 0] = 0.5 * (x[1,0] + x[0,1])
    x[0, -1] = 0.5 * (x[1,-1] + x[0,-2])
    x[-1, 0] = 0.5 * (x[-2,0] + x[-1,1])
    x[-1, -1] = 0.5 * (x[-2,-1] + x[-1,-2])

def lin_solve(b, x, x0, a, c, iter_count, N):
    """
    Solve linear system using Gauss-Seidel:
    x = (x0 + a * sum(neighbors)) / c
    """
    for _ in range(iter_count):
        x[1:-1,1:-1] = (x0[1:-1,1:-1] + a * (
            x[2:,1:-1] + x[:-2,1:-1] + x[1:-1,2:] + x[1:-1,:-2]
        )) / c
        set_bnd(b, x, N)

def diffuse(b, x, x0, diff, dt, iter_count, N):
    a = diff * dt * (N-2) * (N-2)
    lin_solve(b, x, x0, a, 1 + 4 * a, iter_count, N)

def advect(b, d, d0, u, v, dt, N):
    """
    Semi-Lagrangian advection:
    Backtrace from grid cell centers following velocity field.
    """
    dt0 = dt * (N-2)
    # Coordinates of cell centers: i from 1..N-2, j from 1..N-2
    X, Y = np.meshgrid(np.arange(1, N-1), np.arange(1, N-1), indexing='xy')
    x = X - dt0 * u[1:-1,1:-1]
    y = Y - dt0 * v[1:-1,1:-1]

    # Clamp positions to inside the domain
    x = np.clip(x, 0.5, N - 2 + 0.5)
    y = np.clip(y, 0.5, N - 2 + 0.5)

    i0 = x.astype(int)
    j0 = y.astype(int)
    i1 = i0 + 1
    j1 = j0 + 1

    s1 = x - i0
    s0 = 1 - s1
    t1 = y - j0
    t0 = 1 - t1

    # Note: d0 has same padded layout as d; need to offset indices by 0 because i,j already in grid coords
    d[1:-1,1:-1] = (
        s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
        s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
    )
    set_bnd(b, d, N)

def project(u, v, p, div, iter_count, N):
    """
    Make velocity field divergence-free (incompressible):
    Solve for pressure p from divergence, then subtract gradient of p from velocity.
    """
    h = 1.0 / (N-2)
    div[1:-1,1:-1] = -0.5 * h * (u[2:,1:-1] - u[:-2,1:-1] + v[1:-1,2:] - v[1:-1,:-2])
    p.fill(0.0)
    set_bnd(0, div, N)
    set_bnd(0, p, N)

    # Solve ∇² p = div
    lin_solve(0, p, div, 1, 4, iter_count, N)

    # Subtract gradient
    u[1:-1,1:-1] -= 0.5 * (p[2:,1:-1] - p[:-2,1:-1]) / h
    v[1:-1,1:-1] -= 0.5 * (p[1:-1,2:] - p[1:-1,:-2]) / h

    set_bnd(1, u, N)
    set_bnd(2, v, N)

# -----------------------
# Fluid class (holds fields + step)
# -----------------------
class StableFluid2D:
    def __init__(self, N=64, diff=0.0, visc=0.0001, dt=0.1, solver_iters=20):
        """
        N: grid size including 2 boundary padding cells -> internal cells are N-2
        diff: density diffusion
        visc: viscosity (velocity diffusion)
        dt: timestep
        solver_iters: Gauss-Seidel iterations for linear solves
        """
        self.N = N
        self.size = (N, N)
        self.dt = dt
        self.diff = diff
        self.visc = visc
        self.iter = solver_iters

        # Fields: padded arrays (0..N-1 in both dims)
        self.u = np.zeros(self.size)       # horizontal velocity
        self.v = np.zeros(self.size)       # vertical velocity
        self.u_prev = np.zeros(self.size)
        self.v_prev = np.zeros(self.size)

        self.density = np.zeros(self.size)
        self.density_prev = np.zeros(self.size)

        # temporary arrays for projection
        self.p = np.zeros(self.size)
        self.div = np.zeros(self.size)

    def step(self):
        N = self.N
        dt = self.dt

        # Velocity step
        self.u_prev, self.u = self.u, self.u_prev   # swap buffers
        self.v_prev, self.v = self.v, self.v_prev

        # add forces already included directly into u_prev/v_prev by user before step

        # diffuse velocities
        diffuse(1, self.u, self.u_prev, self.visc, dt, self.iter, N)
        diffuse(2, self.v, self.v_prev, self.visc, dt, self.iter, N)

        # advect velocities
        self.u_prev, self.u = self.u, self.u_prev
        self.v_prev, self.v = self.v, self.v_prev
        advect(1, self.u, self.u_prev, self.u_prev, self.v_prev, dt, N)
        advect(2, self.v, self.v_prev, self.u_prev, self.v_prev, dt, N)

        # project to make incompressible
        project(self.u, self.v, self.p, self.div, self.iter, N)

        # Density step
        self.density_prev, self.density = self.density, self.density_prev

        # diffuse density
        diffuse(0, self.density, self.density_prev, self.diff, dt, self.iter, N)

        # advect density with current velocity
        self.density_prev, self.density = self.density, self.density_prev
        advect(0, self.density, self.density_prev, self.u, self.v, dt, N)

    # convenience to add density or velocity sources
    def add_density(self, x, y, amount):
        """x,y should be indices in [1, N-2]"""
        self.density[y, x] += amount

    def add_velocity(self, x, y, amount_u, amount_v):
        self.u[y, x] += amount_u
        self.v[y, x] += amount_v
        
# -----------------------
# Setup fluids
# -----------------------
fluids = []

# Fluid 1: default smoke rising slowly
fluid1 = StableFluid2D(N=66, diff=0.0005, visc=0.00005, dt=0.1)
fluids.append(('Slow Rising', fluid1))

# Fluid 2: very diffusive smoke
fluid2 = StableFluid2D(N=66, diff=0.01, visc=0.00005, dt=0.1)
fluids.append(('Highly Diffusive', fluid2))

# Fluid 3: swirling vortex
fluid3 = StableFluid2D(N=66, diff=0.0005, visc=0.00005, dt=0.1)
fluids.append(('Vortex', fluid3))

# -----------------------
# Visualization
# -----------------------
fig, axes = plt.subplots(1, 3, figsize=(15,5))
ims = []

# Initialize images
for ax, (title, f) in zip(axes, fluids):
    ax.set_title(title)
    ax.axis('off')

images = [ax.imshow(f.density[1:-1,1:-1], origin='lower',
                    interpolation='bilinear', vmin=0, vmax=1.0, cmap='inferno')
          for ax, (_, f) in zip(axes, fluids)]

# -----------------------
# Animation step
# -----------------------
def inject_sources(fluid, pattern, step):
    Nint = fluid.N - 2
    cx, cy = Nint // 2, Nint // 4

    if pattern == 'Slow Rising':
        if step % 5 == 0:
            fluid.add_density(cx, cy, 10.0)
            fluid.add_velocity(cx, cy, 0, 2.0)

    elif pattern == 'Highly Diffusive':
        if step % 3 == 0:
            for dx in range(-2,3):
                for dy in range(-2,3):
                    fluid.add_density(cx+dx, cy+dy, 20.0)
                    fluid.add_velocity(cx+dx, cy+dy, 0, 1.0)

    elif pattern == 'Vortex':
        r = 5
        for dx in range(-r,r+1):
            for dy in range(-r,r+1):
                if dx*dx + dy*dy <= r*r:
                    fluid.add_density(cx+dx, cy+dy, 15.0)
                    fluid.add_velocity(cx+dx, cy+dy, -dy*0.5, dx*0.5)

def update(frame):
    for im, (title, f) in zip(images, fluids):
        inject_sources(f, title, frame)
        f.step()
        im.set_data(f.density[1:-1,1:-1])
    return images

ani = animation.FuncAnimation(fig, update, frames=300, interval=50, blit=False)
plt.show()
