import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.signal import savgol_filter

class EcoEpidemiologicalSolver:
    def __init__(self, params, initial_conditions, T=100, dt=0.01):
        self.p = params
        self.state = np.array(initial_conditions)
        self.T = T
        self.dt = dt
        self.time_steps = int(T / dt)
        self.history = np.zeros((self.time_steps, 3))
        self.time_axis = np.linspace(0, T, self.time_steps)

    def _drift(self, x, y, z):
        rho, l = self.p['rho'], self.p['l']
        phi1, phi2 = self.p['phi1'], self.p['phi2']
        alpha, beta = self.p['alpha'], self.p['beta']
        d, eps = self.p['d'], self.p['epsilon']
        pi_val, h1, h2 = self.p['pi'], self.p['h1'], self.p['h2']

        denom = 1 + l * (1 - rho) * x

        dx = x * (1 - x) - (x * (1 - rho) * (phi1 * y + phi2 * z)) / denom
        dy = (alpha * phi1 * (1 - rho) * x * y) / denom - d * y - eps * y * z - h1 * y
        dz = (beta * phi2 * (1 - rho) * x * z) / denom + eps * y * z - pi_val * z - h2 * z

        return np.array([dx, dy, dz])

    def _diffusion(self, x, y, z):
        return np.array([
            self.p['sigma1'] * x,
            self.p['sigma2'] * y,
            self.p['sigma3'] * z
        ])

    def _diffusion_prime(self, x, y, z):
        return np.array([
            self.p['sigma1'],
            self.p['sigma2'],
            self.p['sigma3']
        ])

    def solve(self):
        curr_state = self.state.copy()
        self.history[0] = curr_state

        dW = np.random.normal(0, np.sqrt(self.dt), (self.time_steps, 3))

        for t in range(1, self.time_steps):
            x, y, z = curr_state
            a = self._drift(x, y, z)
            b = self._diffusion(x, y, z)
            b_prime = self._diffusion_prime(x, y, z)

            milstein_term = 0.5 * b * b_prime * (dW[t]**2 - self.dt)
            next_state = curr_state + a * self.dt + b * dW[t] + milstein_term

            curr_state = np.maximum(next_state, 0)
            self.history[t] = curr_state

        return self.history

params = {
    'rho': 0.3, 'l': 1.0,
    'phi1': 0.5, 'phi2': 0.3,
    'alpha': 0.6, 'beta': 0.4,
    'd': 0.1, 'pi': 0.2, 'epsilon': 0.4,
    'h1': 0.1, 'h2': 0.2,
    'sigma1': 0.01,
    'sigma2': 0.01,
    'sigma3': 0.01
}


solver = EcoEpidemiologicalSolver(params, [0.8, 0.4, 0.2], T=80, dt=0.01)
X_raw = solver.solve()
t = solver.time_axis
dt = solver.dt

X_smooth = savgol_filter(X_raw, window_length=51, polyorder=3, axis=0)

diff_method = ps.SINDyDerivative(
    kind='savitzky_golay',
    left=3,
    right=3,
    order=3
)

feature_library = ps.PolynomialLibrary(degree=2)

optimizer = ps.STLSQ(threshold=0.1)

model = ps.SINDy(feature_library=feature_library,
                 optimizer=optimizer,
                 differentiation_method=diff_method)

model.fit(X_smooth, t=dt)

print("\nOptimized Discovered Model:")
model.print()

X_sim = model.simulate(X_smooth[0], t)

plt.figure(figsize=(12,6))

plt.plot(t, X_raw[:,0], label="True Prey", alpha=0.6)
plt.plot(t, X_sim[:,0], '--', label="SINDy Prey")

plt.plot(t, X_raw[:,1], label="True Susceptible", alpha=0.6)
plt.plot(t, X_sim[:,1], '--', label="SINDy Susceptible")

plt.plot(t, X_raw[:,2], label="True Infected", alpha=0.6)
plt.plot(t, X_sim[:,2], '--', label="SINDy Infected")

plt.xlabel("Time")
plt.ylabel("Population Density")
plt.title("Optimized SINDy vs True Eco-Epidemiological Model")
plt.legend()
plt.grid(True)
plt.show()
