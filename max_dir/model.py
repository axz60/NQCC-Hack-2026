import numpy as np
import math
import matplotlib.pyplot as plt
from pyscf import gto, scf, ao2mo, fci

class Hubbard1D:

    """
    Spinful 1D Hubbard model in a fixed full Fock-space basis.

    Qubit / spin-orbital ordering:
        orbital = 2 * site + spin
        spin = 0 -> up
        spin = 1 -> down
    """

    def __init__(self, nsites: int, nelectrons: tuple, U: float, t: float, periodic: bool = False):

        assert type(nelectrons) == tuple, "nelectrons must be a tuple of (n_up, n_down)"

        self.nsites = nsites
        self.n_electrons = nelectrons
        self.n_up = nelectrons[0]
        self.n_down = nelectrons[1]
        self.U = U
        self.t = t
        self.periodic = periodic

        self.n_spin_orbitals = 2 * nsites
        self.n_qubits = self.n_spin_orbitals
        self.dim = 2 ** self.n_spin_orbitals
        self.up_dim =  math.comb(self.nsites, self.n_up)
        self.down_dim =  math.comb(self.nsites, self.n_down)
        self.spin_dim = self.up_dim * self.down_dim
        

    def count_up(self, state):

        # creates a mask that has 1s in the positions of the up spin orbitals (even indices)
        mask = 0
        for bit in range(0, self.n_spin_orbitals, 2):
            mask |= (1 << bit)

        # siving the state with the mask and counting the number of 1s in the result
        return (state & mask).bit_count()

    def count_down(self, state):

        # creates a mask that has 1s in the positions of the down spin orbitals (odd indices)        

        mask = 0
        for bit in range(1, self.n_spin_orbitals, 2):
            mask |= (1 << bit)

        # siving the state with the mask and counting the number of 1s in the result
        return (state & mask).bit_count()
    

    def basis_states(self):
        
        # If n_electrons is None, return all basis states. Otherwise, return only those states that have the correct number of up and down electrons.
        if self.n_electrons is None:
            return list(range(self.dim))

        # find states that have the correct number of up and down electrons. 
        # This fixes the S_z sector by enforcing fixed N_up and N_down.
        # subspace of the Hilbert space for the given electron configuration.
        return [
            state for state in range(self.dim)
            if self.count_up(state) == self.n_up and self.count_down(state) == self.n_down
            ]

    def orbital(self, site, spin):
        return 2 * site + spin

    def occupation(self, state, orbital):
        return (state >> orbital) & 1

    def fermionic_sign(self, state, p, q):
        """
        Sign for c_p^dagger c_q acting on |state>.
        """
        if p == q:
            return 1

        lo = min(p, q) + 1
        hi = max(p, q)
        n_between = 0

        for r in range(lo, hi):
            n_between += self.occupation(state, r)

        return (-1) ** n_between

    def apply_cdagger_c(self, state, p, q):
        """
        Apply c_p^dagger c_q to a basis state.

        Returns:
            new_state, sign

        or:
            None, 0
        if the operation annihilates the state.
        """
        if self.occupation(state, q) == 0:
            return None, 0

        if self.occupation(state, p) == 1:
            return None, 0

        sign = self.fermionic_sign(state, p, q)

        new_state = state
        new_state ^= (1 << q)
        new_state ^= (1 << p)

        return new_state, sign

    def neighbours(self):
        pairs = [(i, i + 1) for i in range(self.nsites - 1)]

        # Avoid double-counting the same bond for 2-site periodic chain
        if self.periodic and self.nsites > 2:
            pairs.append((self.nsites - 1, 0))

        return pairs

    def hamiltonian(self):

        basis = self.basis_states()
        index = {state: i for i, state in enumerate(basis)}

        H = np.zeros((len(basis), len(basis)), dtype=complex)

        for col, state in enumerate(basis):

            # On-site interaction: U n_up n_down
            for site in range(self.nsites):
                up = self.orbital(site, 0)
                down = self.orbital(site, 1)

                if self.occupation(state, up) and self.occupation(state, down):
                    H[col, col] += self.U

            # Hopping
            for i, j in self.neighbours():
                for spin in [0, 1]:
                    p_i = self.orbital(i, spin)
                    p_j = self.orbital(j, spin)

                    # c_i^dagger c_j
                    new_state, sign = self.apply_cdagger_c(state, p_i, p_j)

                    if new_state is not None and new_state in index:
                        row = index[new_state]
                        H[row, col] += -self.t * sign

                    # c_j^dagger c_i
                    new_state, sign = self.apply_cdagger_c(state, p_j, p_i)

                    if new_state is not None and new_state in index:
                        row = index[new_state]
                        H[row, col] += -self.t * sign

        return H

    def one_two_electron_integrals(self):
        h1 = np.zeros((self.nsites, self.nsites))
        eri = np.zeros((self.nsites, self.nsites, self.nsites, self.nsites))

        for i, j in self.neighbours():
            h1[i, j] += -self.t
            h1[j, i] += -self.t

        for i in range(self.nsites):
            eri[i, i, i, i] = self.U

        return h1, eri

    def ground_state(self):
        H = self.hamiltonian()
        evals, evecs = np.linalg.eigh(H)

        return evals[0], evecs[:, 0]

class Hubbard2D:

    """
    Spinful 1D Hubbard model in a fixed full Fock-space basis.

    Qubit / spin-orbital ordering:
        orbital = 2 * site + spin
        spin = 0 -> up
        spin = 1 -> down
    """

    def __init__(self, plaquette: tuple, nelectrons: tuple, U: float, t: float, periodic: bool = False):

        assert type(nelectrons) == tuple, "nelectrons must be a tuple of (n_up, n_down)"

        self.plaquette = plaquette
        self.Lx = plaquette[0]
        self.Ly = plaquette[1]
        self.nsites = self.Lx * self.Ly
        self.n_electrons = nelectrons
        self.n_up = nelectrons[0]
        self.n_down = nelectrons[1]

        self.U = U
        self.t = t
        self.periodic = periodic

        self.n_spin_orbitals = 2 * self.nsites
        self.n_qubits = self.n_spin_orbitals
        self.dim = 2 ** self.n_spin_orbitals

        self.up_dim = math.comb(self.nsites, self.n_up)
        self.down_dim = math.comb(self.nsites, self.n_down)
        self.spin_dim = self.up_dim * self.down_dim
        

    def count_up(self, state):

        # creates a mask that has 1s in the positions of the up spin orbitals (even indices)
        mask = 0
        for bit in range(0, self.n_spin_orbitals, 2):
            mask |= (1 << bit)

        # siving the state with the mask and counting the number of 1s in the result
        return (state & mask).bit_count()

    def count_down(self, state):

        # creates a mask that has 1s in the positions of the down spin orbitals (odd indices)        

        mask = 0
        for bit in range(1, self.n_spin_orbitals, 2):
            mask |= (1 << bit)

        # siving the state with the mask and counting the number of 1s in the result
        return (state & mask).bit_count()
    

    def basis_states(self):
        
        # If n_electrons is None, return all basis states. Otherwise, return only those states that have the correct number of up and down electrons.
        if self.n_electrons is None:
            return list(range(self.dim))

        # find states that have the correct number of up and down electrons. 
        # This fixes the S_z sector by enforcing fixed N_up and N_down.
        # subspace of the Hilbert space for the given electron configuration.
        return [
            state for state in range(self.dim)
            if self.count_up(state) == self.n_up and self.count_down(state) == self.n_down
            ]

    def orbital(self, site, spin):
        # Help function to get the orbital index from site and spin
        # return odd index for down spin and even index for up spin
        return 2 * site + spin

    def occupation(self, state, orbital):
        # Help function to get the occupation of a given orbital in a given state
        return (state >> orbital) & 1

    def fermionic_sign(self, state, p, q):
        """
        Sign for c_p^dagger c_q acting on |state>.
        """
        if p == q:
            return 1

        lo = min(p, q) + 1
        hi = max(p, q)
        n_between = 0

        for r in range(lo, hi):
            n_between += self.occupation(state, r)

        return (-1) ** n_between

    def apply_cdagger_c(self, state, p, q):
        """
        Apply c_p^dagger c_q to a basis state.

        Returns:
            new_state, sign

        or:
            None, 0
        if the operation annihilates the state.
        """
        if self.occupation(state, q) == 0:
            return None, 0

        if self.occupation(state, p) == 1:
            return None, 0

        sign = self.fermionic_sign(state, p, q)

        new_state = state
        new_state ^= (1 << q)
        new_state ^= (1 << p)

        return new_state, sign

    def site_index(self, x, y):
        return y * self.Lx + x


    def neighbours(self):
        pairs = []

        for y in range(self.Ly):
            for x in range(self.Lx):

                site = self.site_index(x, y)

                # right neighbour
                if x + 1 < self.Lx:
                    right = self.site_index(x + 1, y)
                    pairs.append((site, right))

                # upward neighbour
                if y + 1 < self.Ly:
                    up = self.site_index(x, y + 1)
                    pairs.append((site, up))

        return pairs

    def hamiltonian(self):

        basis = self.basis_states()
        index = {state: i for i, state in enumerate(basis)}

        H = np.zeros((len(basis), len(basis)), dtype=complex)

        for col, state in enumerate(basis):

            # On-site interaction: U n_up n_down
            for site in range(self.nsites):
                up = self.orbital(site, 0)
                down = self.orbital(site, 1)

                if self.occupation(state, up) and self.occupation(state, down):
                    H[col, col] += self.U

            # Hopping
            for i, j in self.neighbours():
                for spin in [0, 1]:
                    p_i = self.orbital(i, spin)
                    p_j = self.orbital(j, spin)

                    # c_i^dagger c_j
                    new_state, sign = self.apply_cdagger_c(state, p_i, p_j)

                    if new_state is not None and new_state in index:
                        row = index[new_state]
                        H[row, col] += -self.t * sign

                    # c_j^dagger c_i
                    new_state, sign = self.apply_cdagger_c(state, p_j, p_i)

                    if new_state is not None and new_state in index:
                        row = index[new_state]
                        H[row, col] += -self.t * sign

        return H
    
    def ground_state(self):
        H = self.hamiltonian()
        evals, evecs = np.linalg.eigh(H)

        return evals[0], evecs[:, 0]

    def one_two_electron_integrals(self):
        """
        Spatial-orbital one- and two-electron integrals for the 2D Hubbard model.
        Compatible with one_particle(), two_particle(), observable().
        """

        norb = self.nsites

        h1 = np.zeros((norb, norb), dtype=float)
        h2 = np.zeros((norb, norb, norb, norb), dtype=float)

        # hopping
        for i, j in self.neighbours():
            h1[i, j] += -self.t
            h1[j, i] += -self.t

        # onsite Hubbard U: (ii|ii) = U
        for i in range(self.nsites):
            h2[i, i, i, i] = self.U

        return h1, h2


