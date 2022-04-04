import sys
import numpy as np

# Class implementing the Fourier representation
# of the 1-dimensional single-particle Hamilton operator.
# Implementation follows:
# https://www.physik.hu-berlin.de/de/com/teachingandseminars/previousCPII
class Hamiltonian:
    def __init__(self, length, n_steps):
        if np.mod(n_steps, 2) != 1:
            sys.exit("Error class Hamiltonian: Need odd number of steps")
        self.n_steps = n_steps
        self.length = length
        self.stepsize = length / n_steps
        self.m = (n_steps - 1) / 2
        self.x_values = self.stepsize * np.linspace(-self.m, self.m, self.n_steps)
        self.build_kinetic_hamiltonian()

    def build_hamiltonian(self, potential_func):
        self.build_potential_hamiltonian(potential_func)
        self.hamiltonian = self.hamilton_kinetic + self.hamilton_potential

    def build_potential_hamiltonian(self, V_func):
        self.hamilton_potential = np.diag(V_func(self.x_values))

    def build_kinetic_hamiltonian(self):
        self.momenta = (2 * np.pi / self.length) * np.linspace(
            -self.m, self.m, self.n_steps
        )
        H = np.zeros((self.n_steps, self.n_steps))

        # Matrix A with elements A_kl = x_k - x_l
        x_mat = np.outer(self.x_values, np.ones(self.n_steps))
        A_mat = x_mat - np.transpose(x_mat)
        for i in range(self.n_steps):
            H += self.momenta[i] ** 2 * np.cos(self.momenta[i] * A_mat)
        self.hamilton_kinetic = 0.5 / self.n_steps * H

    def diagonalize(self):
        eigvals, eigvecs = np.linalg.eig(self.hamiltonian)

        # Sort eigenvalues by energy
        index = np.argsort(eigvals)
        self.eigvecs = eigvecs[:, index]
        self.eigvals = eigvals[index]

    def calc_scatter_phases(self):
        # Keep only scattering solutions with positive energy
        eigvals_scatter = np.delete(self.eigvals, np.where(self.eigvals < 0))
        eigvecs_scatter = np.delete(self.eigvecs, np.where(self.eigvals < 0), axis=1)

        # Sort wavefunctions by parity
        eigvecs_rev = eigvecs_scatter[::-1]
        ids_neg = np.where(
            np.all(np.isclose(eigvecs_rev, -eigvecs_scatter, atol=1e-5), axis=0)
        )[0]
        ids_pos = np.where(
            np.all(np.isclose(eigvecs_rev, eigvecs_scatter, atol=1e-5), axis=0)
        )[0]

        # Compute values corresponding to positive or negative parity
        kvals_pos = np.sqrt(2 * eigvals_scatter[ids_pos] + 0j)
        kvals_neg = np.sqrt(2 * eigvals_scatter[ids_neg] + 0j)
        delta_pos = np.real(-1j * np.log(np.exp(-1j * kvals_pos * self.length)) / 2)
        delta_neg = np.real(-1j * np.log(np.exp(-1j * kvals_neg * self.length)) / 2)

        return delta_neg, delta_pos, kvals_neg, kvals_pos
