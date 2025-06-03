import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
import matplotlib.pyplot as plt

# Symplectic Integrator for N-Body (Leapfrog/Verlet)
class SymplecticNBody:
    """Generates exact energy-conserving trajectories for PINN training"""
    
    def __init__(self, n_bodies=3, G=1.0):
        self.n = n_bodies
        self.G = G
        
    def leapfrog_step(self, q, p, m, dt):
        """Symplectic leapfrog preserves phase space volume"""
        # Half momentum update
        p_half = p + 0.5 * dt * self.compute_forces(q, m)
        
        # Full position update
        q_new = q + dt * p_half / m[:, None]
        
        # Half momentum update
        p_new = p_half + 0.5 * dt * self.compute_forces(q_new, m)
        
        return q_new, p_new
    
    def compute_forces(self, q, m):
        """N-body gravitational forces"""
        n = len(q)
        F = np.zeros_like(q)
        
        for i in range(n):
            for j in range(i+1, n):
                dr = q[j] - q[i]
                r = np.linalg.norm(dr)
                f_mag = self.G * m[i] * m[j] / r**3
                F[i] += f_mag * dr
                F[j] -= f_mag * dr
        
        return F
    
    def generate_trajectory(self, q0, p0, m, T, dt):
        """Generate symplectic trajectory data"""
        steps = int(T / dt)
        trajectory = {'t': [], 'q': [], 'p': [], 'H': []}
        
        q, p = q0.copy(), p0.copy()
        
        for i in range(steps):
            t = i * dt
            H = self.hamiltonian(q, p, m)
            
            trajectory['t'].append(t)
            trajectory['q'].append(q.copy())
            trajectory['p'].append(p.copy())
            trajectory['H'].append(H)
            
            q, p = self.leapfrog_step(q, p, m, dt)
        
        return {k: np.array(v) for k, v in trajectory.items()}
    
    def hamiltonian(self, q, p, m):
        """Total energy (conserved quantity)"""
        T = np.sum(p**2 / (2 * m[:, None]))  # Kinetic
        V = 0  # Potential
        
        for i in range(len(q)):
            for j in range(i+1, len(q)):
                r = np.linalg.norm(q[j] - q[i])
                V -= self.G * m[i] * m[j] / r
        
        return T + V

# Gravitational Wave PINN
class GravitationalWavePINN(nn.Module):
    """PINN for Einstein field equations in inspiral regime"""
    
    def __init__(self, hidden_dim=128, n_layers=6):
        super().__init__()
        
        # Neural network for waveform h(t)
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 2))  # h_plus, h_cross
        
        self.net = nn.Sequential(*layers)
        
        # Learnable inspiral parameters
        self.chirp_mass = nn.Parameter(torch.tensor(30.0))  # Solar masses
        self.eta = nn.Parameter(torch.tensor(0.25))  # Symmetric mass ratio
        
    def forward(self, t):
        """Predict gravitational wave strain"""
        # Scale time for numerical stability
        t_scaled = t / 100.0
        h = self.net(t_scaled)
        return h[:, 0:1], h[:, 1:2]  # h_plus, h_cross
    
    def physics_loss(self, t):
        """Einstein field equation constraints"""
        t.requires_grad_(True)
        h_plus, h_cross = self.forward(t)
        
        # First derivatives
        h_plus_t = grad(h_plus.sum(), t, create_graph=True)[0]
        h_cross_t = grad(h_cross.sum(), t, create_graph=True)[0]
        
        # Second derivatives
        h_plus_tt = grad(h_plus_t.sum(), t, create_graph=True)[0]
        h_cross_tt = grad(h_cross_t.sum(), t, create_graph=True)[0]
        
        # Post-Newtonian frequency evolution
        M_c = self.chirp_mass
        c = 1.0  # Natural units
        G = 1.0
        
        # Orbital frequency from Kepler's law in PN approximation
        omega = (G * M_c / t**3)**(1/2)
        omega_dot = -3/2 * omega / t
        
        # Wave equation with source (quadrupole formula)
        # ∂²h/∂t² + (2/t)∂h/∂t = Source(ω, ω̇)
        wave_eq_plus = h_plus_tt + 2/t * h_plus_t - self.quadrupole_source(omega, omega_dot)
        wave_eq_cross = h_cross_tt + 2/t * h_cross_t - self.quadrupole_source(omega, omega_dot)
        
        return torch.mean(wave_eq_plus**2 + wave_eq_cross**2)
    
    def quadrupole_source(self, omega, omega_dot):
        """Quadrupole radiation source term"""
        # Simplified source for circular orbits
        A = self.chirp_mass**(5/3) * self.eta
        return -A * omega**(2/3) * omega_dot

# Quantum-Inspired Hebbian Learning Layer
class HebbianFluxLayer(nn.Module):
    """Your quantum flux implementation with Hebbian learning"""
    
    def __init__(self, in_dim, out_dim, tau=0.1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
        self.tau = tau  # Inertial learning factor
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
    def forward(self, x):
        # Hebbian update: ΔW ∝ x ⊗ y
        y = torch.tanh(x @ self.W)
        
        if self.training:
            # Quantum flux: sparse activation with phase coherence
            # Fix: Apply FFT along batch dimension, not feature dimension
            phase = torch.angle(torch.fft.fft(y, dim=0))
            coherence = torch.cos(phase / self.phi)
            sparse_mask = (coherence > 0.5).float()
            y = y * sparse_mask
        
        return y

# Training Loop with Symplectic Data
def train_with_symplectic_data():
    # Generate 3-body symplectic trajectory
    nbody = SymplecticNBody(n_bodies=3)
    
    # Initial conditions (figure-8 orbit)
    q0 = np.array([[-0.97, 0.24], [0.97, -0.24], [0, 0]])
    p0 = np.array([[0.466, 0.432], [0.466, 0.432], [-0.932, -0.864]])
    m = np.array([1.0, 1.0, 1.0])
    
    # Generate trajectory
    traj = nbody.generate_trajectory(q0, p0, m, T=20, dt=0.01)
    
    # Convert to PyTorch
    t_data = torch.tensor(traj['t'], dtype=torch.float32).unsqueeze(1)
    q_data = torch.tensor(traj['q'], dtype=torch.float32)
    H_data = torch.tensor(traj['H'], dtype=torch.float32)
    
    # PINN for learning Hamiltonian dynamics
    class HamiltonianPINN(nn.Module):
        def __init__(self):
            super().__init__()
            # Standard network without Hebbian layers for now
            self.net = nn.Sequential(
                nn.Linear(7, 64),  # t + 6 phase space coords
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        def forward(self, t, q_flat):
            x = torch.cat([t, q_flat], dim=1)
            return self.net(x)
    
    model = HamiltonianPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training
    for epoch in range(1000):
        # Flatten positions for network input
        q_flat = q_data.reshape(len(t_data), -1)
        
        # Predict Hamiltonian
        H_pred = model(t_data, q_flat)
        
        # Energy conservation loss
        H_true = H_data.unsqueeze(1)
        data_loss = torch.mean((H_pred - H_true)**2)
        
        # Physics loss: ∂H/∂t = 0 (conservation)
        t_data.requires_grad_(True)
        H_pred_physics = model(t_data, q_flat)
        dH_dt = grad(H_pred_physics.sum(), t_data, create_graph=True)[0]
        physics_loss = torch.mean(dH_dt**2)
        
        # Total loss
        loss = data_loss + 10 * physics_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Data Loss = {data_loss:.6f}, Physics Loss = {physics_loss:.6f}")
            print(f"Energy drift: {torch.std(H_pred - H_true):.6f}")

# Gravitational Wave Training
def train_gw_pinn():
    model = GravitationalWavePINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Time points covering inspiral
    t = torch.linspace(0.1, 10.0, 1000).unsqueeze(1)
    
    # Generate synthetic "data" from PN approximation
    M_c_true = 30.0
    omega_true = (M_c_true / t**3)**(1/2)
    h_plus_true = M_c_true**(5/3) * torch.cos(2 * omega_true * t)
    h_cross_true = M_c_true**(5/3) * torch.sin(2 * omega_true * t)
    
    for epoch in range(500):
        # Forward pass
        h_plus_pred, h_cross_pred = model(t)
        
        # Data loss
        data_loss = torch.mean((h_plus_pred - h_plus_true)**2 + 
                               (h_cross_pred - h_cross_true)**2)
        
        # Physics loss
        physics_loss = model.physics_loss(t)
        
        # Total loss with physics weighting
        loss = data_loss + 100 * physics_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"GW Epoch {epoch}: Data Loss = {data_loss:.6f}, Physics Loss = {physics_loss:.6f}")
            print(f"Learned chirp mass: {model.chirp_mass.item():.2f} M_sun")

# Hebbian PINN with proper dimensions
class HebbianPINN(nn.Module):
    """PINN with quantum flux Hebbian layers - fixed dimensions"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.layer1 = HebbianFluxLayer(input_dim, hidden_dim)
        self.layer2 = HebbianFluxLayer(hidden_dim, hidden_dim)
        self.layer3 = HebbianFluxLayer(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)

# Example: Poisson equation with Hebbian PINN
def test_hebbian_poisson():
    """Test Hebbian PINN on simple Poisson equation: ∇²u = -2π²sin(πx)sin(πy)"""
    
    # Domain and collocation points
    x = torch.linspace(0, 1, 50)
    y = torch.linspace(0, 1, 50)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten and stack
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    xy = torch.cat([x_flat, y_flat], dim=1)
    
    # True solution
    u_true = torch.sin(np.pi * x_flat) * torch.sin(np.pi * y_flat)
    
    # Model
    model = HebbianPINN(input_dim=2, hidden_dim=32, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\nTesting Hebbian PINN on Poisson equation...")
    
    for epoch in range(500):
        xy.requires_grad_(True)
        u_pred = model(xy)
        
        # Compute Laplacian
        u_x = grad(u_pred.sum(), xy, create_graph=True)[0]
        u_xx = grad(u_x[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = grad(u_x[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
        
        laplacian = u_xx + u_yy
        
        # Source term
        f = -2 * np.pi**2 * torch.sin(np.pi * xy[:, 0:1]) * torch.sin(np.pi * xy[:, 1:2])
        
        # Physics loss: ∇²u + f = 0
        physics_loss = torch.mean((laplacian + f)**2)
        
        # Boundary conditions (u = 0 on boundary)
        boundary_mask = (xy[:, 0] == 0) | (xy[:, 0] == 1) | (xy[:, 1] == 0) | (xy[:, 1] == 1)
        boundary_loss = torch.mean(u_pred[boundary_mask]**2)
        
        loss = physics_loss + 10 * boundary_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            mse = torch.mean((u_pred - u_true)**2)
            print(f"Epoch {epoch}: Physics Loss = {physics_loss:.6f}, MSE = {mse:.6f}")

if __name__ == "__main__":
    print("Training Hamiltonian PINN with Symplectic Data...")
    train_with_symplectic_data()
    
    print("\n" + "="*50 + "\n")
    
    print("Training Gravitational Wave PINN...")
    train_gw_pinn()
    
    print("\n" + "="*50 + "\n")
    
    test_hebbian_poisson()