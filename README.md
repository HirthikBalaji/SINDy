# Eco-Epidemiological SDE Solver with SINDy Discovery

A Python framework for simulating and **automatically discovering the governing equations** of a stochastic eco-epidemiological system using the **Milstein scheme** and **Sparse Identification of Nonlinear Dynamics (SINDy)**.

---

## 📌 Overview

This project models a three-compartment population system — **Prey (x)**, **Susceptible Predators (y)**, and **Infected Predators (z)** — governed by nonlinear stochastic differential equations (SDEs). The system incorporates:

- **Disease transmission** between predator compartments
- **Harvesting** of susceptible and infected predators
- **Multiplicative noise** via the **Milstein scheme** for higher-order SDE integration
- **Data-driven equation recovery** via **PySINDy** with Savitzky-Golay smoothing

---

## 🧮 Mathematical Model

The deterministic drift of the system is:

$$\frac{dx}{dt} = x(1 - x) - \frac{x(1-\rho)(\phi_1 y + \phi_2 z)}{1 + l(1-\rho)x}$$

$$\frac{dy}{dt} = \frac{\alpha \phi_1 (1-\rho) x y}{1 + l(1-\rho)x} - dy - \varepsilon yz - h_1 y$$

$$\frac{dz}{dt} = \frac{\beta \phi_2 (1-\rho) x z}{1 + l(1-\rho)x} + \varepsilon yz - \pi z - h_2 z$$

With multiplicative noise terms $\sigma_i \cdot X_i \, dW_i$ integrated using the **Milstein correction**.

---

## 🗂️ Project Structure

```
├── model.py       # Main solver, SINDy pipeline, and plotting
└── README.md
```

---

## ⚙️ Installation

```bash
pip install numpy matplotlib pysindy scipy
```

> Python 3.8+ recommended.

---

## 🚀 Usage

```python
from eco_epid_sindy import EcoEpidemiologicalSolver
import pysindy as ps
from scipy.signal import savgol_filter

params = {
    'rho': 0.3, 'l': 1.0,
    'phi1': 0.5, 'phi2': 0.3,
    'alpha': 0.6, 'beta': 0.4,
    'd': 0.1, 'pi': 0.2, 'epsilon': 0.4,
    'h1': 0.1, 'h2': 0.2,
    'sigma1': 0.01, 'sigma2': 0.01, 'sigma3': 0.01
}

solver = EcoEpidemiologicalSolver(params, [0.8, 0.4, 0.2], T=80, dt=0.01)
X_raw = solver.solve()
```

Run the full script to simulate, identify equations, and visualize:

```bash
python eco_epid_sindy.py
```

---

## 🔬 Pipeline

```
Stochastic SDE Simulation (Milstein)
            ↓
   Savitzky-Golay Smoothing
            ↓
 SINDy with Polynomial Library (degree=2)
            ↓
  STLSQ Sparse Optimizer (threshold=0.1)
            ↓
  Discovered Equations + Simulation
```

---

## 📊 Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `rho` | ρ | Prey refuge fraction |
| `l` | l | Crowding / interference coefficient |
| `phi1`, `phi2` | φ₁, φ₂ | Predation rates (susceptible, infected) |
| `alpha`, `beta` | α, β | Conversion efficiencies |
| `d` | d | Natural death rate of susceptible predators |
| `epsilon` | ε | Disease transmission rate |
| `pi` | π | Death rate of infected predators |
| `h1`, `h2` | h₁, h₂ | Harvesting rates |
| `sigma1–3` | σ₁₋₃ | Noise intensities |

---

## 📈 Output

- **Console**: Discovered sparse governing equations printed by PySINDy
- **Plot**: True stochastic trajectories vs. SINDy-simulated trajectories for all three compartments


---

## 🧠 Key Techniques

- **Milstein Scheme**: Second-order SDE integrator correcting for multiplicative noise via the $b \cdot b'$ term, reducing strong approximation error vs. Euler-Maruyama
- **Savitzky-Golay Filter**: Polynomial smoothing applied before differentiation to suppress stochastic noise
- **SINDy (Sparse Identification of Nonlinear Dynamics)**: Constructs a polynomial feature library and uses STLSQ (Sequential Thresholded Least Squares) to find the sparsest equation set that fits the data

---

## 📚 References

- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*. PNAS.
- Kloeden, P. E., & Platen, E. (1992). *Numerical Solution of Stochastic Differential Equations*. Springer.
- de Silva, B. et al. (2020). *PySINDy: A Python package for the sparse identification of nonlinear dynamical systems*. JOSS.

---

## 📄 License

MIT License. See `LICENSE` for details.
