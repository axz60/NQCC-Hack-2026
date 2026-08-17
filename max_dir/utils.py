import numpy as np

class LehmannGF:
    
    """
    Exact Lehmann Green's function and spectral function calculator.

    Requires model objects with:
        model.hamiltonian()
        model.basis_states()
        model.occupation(state, orbital)
        model.orbital(site, spin)
    """

    def __init__(self, model_N, model_N_plus, model_N_minus):
        self.model_N = model_N
        self.model_N_plus = model_N_plus
        self.model_N_minus = model_N_minus

        self._diagonalize_models()

    def _diagonalize_models(self):
        # N sector
        self.basis_N = self.model_N.basis_states()
        self.H_N = self.model_N.hamiltonian()
        self.E_N, self.V_N = np.linalg.eigh(self.H_N)

        self.E0 = self.E_N[0]
        self.psi0 = self.V_N[:, 0]

        # N + 1 sector
        self.basis_plus = self.model_N_plus.basis_states()
        self.H_plus = self.model_N_plus.hamiltonian()
        self.E_plus, self.V_plus = np.linalg.eigh(self.H_plus)

        # N - 1 sector
        self.basis_minus = self.model_N_minus.basis_states()
        self.H_minus = self.model_N_minus.hamiltonian()
        self.E_minus, self.V_minus = np.linalg.eigh(self.H_minus)

    def _annihilate_vector(self, vec, orbital):
        """
        Apply c_orbital to a vector in the N sector.
        Result lives in the N-1 sector.
        """
        index_minus = {state: i for i, state in enumerate(self.basis_minus)}
        out = np.zeros(len(self.basis_minus), dtype=complex)

        for col, state in enumerate(self.basis_N):
            amp = vec[col]

            if abs(amp) < 1e-14:
                continue

            if self.model_N.occupation(state, orbital) == 0:
                continue

            n_before = sum(
                self.model_N.occupation(state, r)
                for r in range(orbital)
            )

            sign = (-1) ** n_before
            new_state = state ^ (1 << orbital)

            if new_state in index_minus:
                row = index_minus[new_state]
                out[row] += sign * amp

        return out

    def _create_vector(self, vec, orbital):
        """
        Apply c_orbital^dagger to a vector in the N sector.
        Result lives in the N+1 sector.
        """
        index_plus = {state: i for i, state in enumerate(self.basis_plus)}
        out = np.zeros(len(self.basis_plus), dtype=complex)

        for col, state in enumerate(self.basis_N):
            amp = vec[col]

            if abs(amp) < 1e-14:
                continue

            if self.model_N.occupation(state, orbital) == 1:
                continue

            n_before = sum(
                self.model_N.occupation(state, r)
                for r in range(orbital)
            )

            sign = (-1) ** n_before
            new_state = state ^ (1 << orbital)

            if new_state in index_plus:
                row = index_plus[new_state]
                out[row] += sign * amp

        return out

    def green_function(self, site, spin, omega_grid, eta=0.05):
        """
        Compute retarded zero-temperature Green's function:

            G(omega) = G_particle(omega) + G_hole(omega)

        for orbital p = 2*site + spin.
        """
        p = self.model_N.orbital(site, spin)

        particle_vec = self._create_vector(self.psi0, p)
        hole_vec = self._annihilate_vector(self.psi0, p)

        G = np.zeros_like(omega_grid, dtype=complex)

        # Particle/addition part
        for m in range(len(self.E_plus)):
            psi_m = self.V_plus[:, m]

            weight = abs(np.vdot(psi_m, particle_vec)) ** 2
            pole = self.E_plus[m] - self.E0

            G += weight / (omega_grid - pole + 1j * eta)

        # Hole/removal part
        for m in range(len(self.E_minus)):
            psi_m = self.V_minus[:, m]

            weight = abs(np.vdot(psi_m, hole_vec)) ** 2
            pole = self.E_minus[m] - self.E0

            G += weight / (omega_grid + pole + 1j * eta)

        return G

    def spectral_function(self, site, spin, omega_grid, eta=0.05):
        G = self.green_function(site, spin, omega_grid, eta)
        A = -(1 / np.pi) * np.imag(G)

        return A

    def green_and_spectral_function(self, site, spin, omega_grid, eta=0.05):
        G = self.green_function(site, spin, omega_grid, eta)
        A = -(1 / np.pi) * np.imag(G)

        return G, A

    def spin_summed_spectral_function(self, site, omega_grid, eta=0.05):
        A_up = self.spectral_function(site, spin=0, omega_grid=omega_grid, eta=eta)
        A_down = self.spectral_function(site, spin=1, omega_grid=omega_grid, eta=eta)

        return A_up + A_down

    def spectral_weights(self, site, spin):

        p = self.model_N.orbital(site, spin)

        particle_vec = self._create_vector(self.psi0, p)
        hole_vec = self._annihilate_vector(self.psi0, p)

        particle_poles = []
        particle_weights = []

        hole_poles = []
        hole_weights = []

        # particle sector
        for m in range(len(self.E_plus)):

            psi_m = self.V_plus[:, m]

            weight = abs(np.vdot(psi_m, particle_vec)) ** 2
            pole = self.E_plus[m] - self.E0

            particle_poles.append(pole)
            particle_weights.append(weight)

        # hole sector
        for m in range(len(self.E_minus)):

            psi_m = self.V_minus[:, m]

            weight = abs(np.vdot(psi_m, hole_vec)) ** 2
            pole = self.E_minus[m] - self.E0

            hole_poles.append(pole)
            hole_weights.append(weight)

        return (
            np.array(particle_poles),
            np.array(particle_weights),
            np.array(hole_poles),
            np.array(hole_weights),
        )

    def spectral_sum_rule(self, site, spin):
        _, Wp, _, Wh = self.spectral_weights(site, spin)
        return np.sum(Wp) + np.sum(Wh)      
        
        
def fci_spectral_moments(
    model,
    nmom=10,
):
    """
    Compute exact FCI hole and particle spectral moments
    for Hubbard1D or Hubbard2D.

    The input model is the N-electron model.

    Moments:

        T_h[n,i,j]
        =
        sum_sigma
        <Psi0|
            c^dag_{i,sigma}
            (E0 - H_{N-1})^n
            c_{j,sigma}
        |Psi0>

        T_p[n,i,j]
        =
        sum_sigma
        <Psi0|
            c_{i,sigma}
            (H_{N+1} - E0)^n
            c^dag_{j,sigma}
        |Psi0>

    Parameters
    ----------
    model
        Hubbard1D or Hubbard2D instance.

    nmom : int
        Maximum moment order.

    Returns
    -------
    dict
        Ground-state energy and spin-resolved /
        spin-summed moments.
    """

    # ==========================================================
    # Helpers
    # ==========================================================

    def build_sector_model(nelectrons):
        """
        Reconstruct the same Hubbard model in another
        (N_up, N_down) sector.
        """

        cls = type(model)

        # Hubbard2D
        if hasattr(model, "plaquette"):

            return cls(
                plaquette=model.plaquette,
                nelectrons=nelectrons,
                U=model.U,
                t=model.t,
                periodic=model.periodic,
            )

        # Hubbard1D
        return cls(
            nsites=model.nsites,
            nelectrons=nelectrons,
            U=model.U,
            t=model.t,
            periodic=model.periodic,
        )


    def apply_annihilation(
        psi,
        orbital,
        basis_from,
        basis_to,
    ):
        """
        Apply c_orbital to a vector.

        basis_from : N-particle basis
        basis_to   : N-1-particle basis
        """

        target_index = {
            state: i
            for i, state in enumerate(basis_to)
        }

        out = np.zeros(
            len(basis_to),
            dtype=complex,
        )

        for col, state in enumerate(basis_from):

            amp = psi[col]

            if abs(amp) < 1e-14:
                continue

            # Orbital must be occupied.
            if model.occupation(state, orbital) == 0:
                continue

            # Jordan-Wigner / fermionic sign.
            n_before = sum(
                model.occupation(state, r)
                for r in range(orbital)
            )

            sign = (-1) ** n_before

            new_state = (
                state ^ (1 << orbital)
            )

            row = target_index.get(
                new_state
            )

            if row is not None:
                out[row] += sign * amp

        return out


    def apply_creation(
        psi,
        orbital,
        basis_from,
        basis_to,
    ):
        """
        Apply c^dag_orbital to a vector.

        basis_from : N-particle basis
        basis_to   : N+1-particle basis
        """

        target_index = {
            state: i
            for i, state in enumerate(basis_to)
        }

        out = np.zeros(
            len(basis_to),
            dtype=complex,
        )

        for col, state in enumerate(basis_from):

            amp = psi[col]

            if abs(amp) < 1e-14:
                continue

            # Orbital must be empty.
            if model.occupation(state, orbital) == 1:
                continue

            n_before = sum(
                model.occupation(state, r)
                for r in range(orbital)
            )

            sign = (-1) ** n_before

            new_state = (
                state ^ (1 << orbital)
            )

            row = target_index.get(
                new_state
            )

            if row is not None:
                out[row] += sign * amp

        return out


    # ==========================================================
    # N-electron FCI ground state
    # ==========================================================

    basis_N = model.basis_states()
    H_N = model.hamiltonian()

    evals, evecs = np.linalg.eigh(
        H_N
    )

    E0 = float(
        np.real(evals[0])
    )

    psi0 = evecs[:, 0]

    nsites = model.nsites

    n_up = model.n_up
    n_down = model.n_down

    shape = (
        nmom + 1,
        nsites,
        nsites,
    )

    hole_up = np.zeros(
        shape,
        dtype=complex,
    )

    hole_down = np.zeros(
        shape,
        dtype=complex,
    )

    particle_up = np.zeros(
        shape,
        dtype=complex,
    )

    particle_down = np.zeros(
        shape,
        dtype=complex,
    )

    # ==========================================================
    # Process one spin sector
    # ==========================================================

    for spin in (0, 1):

        n_spin = (
            n_up
            if spin == 0
            else n_down
        )

        # ======================================================
        # Hole / N-1 sector
        # ======================================================

        if n_spin > 0:

            if spin == 0:
                nelec_minus = (
                    n_up - 1,
                    n_down,
                )
            else:
                nelec_minus = (
                    n_up,
                    n_down - 1,
                )

            model_minus = (
                build_sector_model(
                    nelec_minus
                )
            )

            basis_minus = (
                model_minus.basis_states()
            )

            H_minus = (
                model_minus.hamiltonian()
            )

            # ----------------------------------------------
            # |h_j^(0)> = c_j |Psi0>
            # ----------------------------------------------

            hole_seed = []

            for j in range(nsites):

                orbital = model.orbital(
                    j,
                    spin,
                )

                seed = apply_annihilation(
                    psi0,
                    orbital,
                    basis_N,
                    basis_minus,
                )

                hole_seed.append(seed)

            hole_vec = [
                x.copy()
                for x in hole_seed
            ]

            # ----------------------------------------------
            # Krylov recursion
            # ----------------------------------------------

            for n in range(nmom + 1):

                for i in range(nsites):
                    for j in range(nsites):

                        value = np.vdot(
                            hole_seed[i],
                            hole_vec[j],
                        )

                        if spin == 0:
                            hole_up[
                                n, i, j
                            ] = value

                        else:
                            hole_down[
                                n, i, j
                            ] = value

                if n == nmom:
                    break

                for j in range(nsites):

                    hole_vec[j] = (
                        E0 * hole_vec[j]
                        - H_minus @ hole_vec[j]
                    )

        # ======================================================
        # Particle / N+1 sector
        # ======================================================

        if n_spin < nsites:

            if spin == 0:
                nelec_plus = (
                    n_up + 1,
                    n_down,
                )
            else:
                nelec_plus = (
                    n_up,
                    n_down + 1,
                )

            model_plus = (
                build_sector_model(
                    nelec_plus
                )
            )

            basis_plus = (
                model_plus.basis_states()
            )

            H_plus = (
                model_plus.hamiltonian()
            )

            # ----------------------------------------------
            # |p_j^(0)> = c_j^dag |Psi0>
            # ----------------------------------------------

            particle_seed = []

            for j in range(nsites):

                orbital = model.orbital(
                    j,
                    spin,
                )

                seed = apply_creation(
                    psi0,
                    orbital,
                    basis_N,
                    basis_plus,
                )

                particle_seed.append(seed)

            particle_vec = [
                x.copy()
                for x in particle_seed
            ]

            # ----------------------------------------------
            # Krylov recursion
            # ----------------------------------------------

            for n in range(nmom + 1):

                for i in range(nsites):
                    for j in range(nsites):

                        value = np.vdot(
                            particle_seed[i],
                            particle_vec[j],
                        )

                        if spin == 0:
                            particle_up[
                                n, i, j
                            ] = value

                        else:
                            particle_down[
                                n, i, j
                            ] = value

                if n == nmom:
                    break

                for j in range(nsites):

                    particle_vec[j] = (
                        H_plus @ particle_vec[j]
                        - E0 * particle_vec[j]
                    )

    # ==========================================================
    # Spin-summed moments
    # ==========================================================

    hole = (
        hole_up
        + hole_down
    )

    particle = (
        particle_up
        + particle_down
    )

    return {
        "E0": E0,
        "psi0": psi0,

        "hole_moments_up":
            np.real_if_close(
                hole_up
            ),

        "hole_moments_down":
            np.real_if_close(
                hole_down
            ),

        "particle_moments_up":
            np.real_if_close(
                particle_up
            ),

        "particle_moments_down":
            np.real_if_close(
                particle_down
            ),

        "hole_moments":
            np.real_if_close(
                hole
            ),

        "particle_moments":
            np.real_if_close(
                particle
            ),
    }

def greens_function_from_moments(
    hole_moments,
    particle_moments,
    omega_grid,
    one_body_integrals=None,
    constant_energy=0.0,
    eta=0.05,
    nblocks=None,
    threshold=1e-10,
    verbose=False,
):
    """
    Reconstruct a Green's function from arbitrary
    hole and particle spectral moments.

    The moments may come from:

        - exact FCI
        - classical statevector
        - VQE
        - shot-based simulation
        - quantum hardware

    Also calculates the Galitskii-Migdal energy if
    one_body_integrals are provided.

    Parameters
    ----------
    hole_moments : ndarray
        Shape:
            (nmom + 1, norb, norb)

    particle_moments : ndarray
        Same shape as hole_moments.

    omega_grid : ndarray
        Real-frequency grid.

    one_body_integrals : ndarray or None
        h_ij used for the Galitskii-Migdal energy.

        For Hubbard:
            model.one_two_electron_integrals()[0]

        For a CAS calculation:
            effective CAS one-body integrals.

    constant_energy : float
        Nuclear/frozen-core/etc. constant.

        For Hubbard:
            normally 0.

    eta : float
        Lorentzian broadening.

    nblocks : int or None
        Number of block-lancsoz levels.

    threshold : float
        Eigenvalue cutoff for the block overlap matrix.

    Returns
    -------
    dict
        Green's functions, spectral function, poles,
        residues and GM energy.
    """

    hole_moments = np.asarray(
        hole_moments,
        dtype=complex,
    )

    particle_moments = np.asarray(
        particle_moments,
        dtype=complex,
    )

    omega_grid = np.asarray(
        omega_grid,
        dtype=float,
    )

    # ==========================================================
    # Validation
    # ==========================================================

    if (
        hole_moments.ndim != 3
        or particle_moments.ndim != 3
    ):
        raise ValueError(
            "Moments must have shape "
            "(nmom + 1, norb, norb)."
        )

    if (
        hole_moments.shape
        != particle_moments.shape
    ):
        raise ValueError(
            "Hole and particle moments "
            "must have the same shape."
        )

    nmom_plus_1, norb, norb_2 = (
        hole_moments.shape
    )

    if norb != norb_2:
        raise ValueError(
            "Moment matrices must be square."
        )

    # ==========================================================
    # Helper
    # ==========================================================

    def hermitize(A):
        return 0.5 * (
            A + A.conj().T
        )

    # ==========================================================
    # Block-lancsoz auxiliary reconstruction
    # ==========================================================

    def block_lancsoz(moments):

        moments = np.asarray(
            moments,
            dtype=complex,
        )

        nmom_available = (
            moments.shape[0]
        )

        if nblocks is None:
            nb = (
                nmom_available // 2
            )
        else:
            nb = int(nblocks)

        if nb < 1:
            raise ValueError(
                "nblocks must be >= 1."
            )

        required = (
            2 * nb
        )

        if nmom_available < required:

            raise ValueError(
                f"Need moments through order "
                f"{required - 1} for "
                f"nblocks={nb}, but only "
                f"{nmom_available - 1} are available."
            )

        dim = nb * norb

        S = np.zeros(
            (dim, dim),
            dtype=complex,
        )

        T = np.zeros(
            (dim, dim),
            dtype=complex,
        )

        # --------------------------------------------------
        # Block-lancsoz matrices
        #
        # S_ab = M_{a+b}
        # T_ab = M_{a+b+1}
        # --------------------------------------------------

        for a in range(nb):
            for b in range(nb):

                rows = slice(
                    a * norb,
                    (a + 1) * norb,
                )

                cols = slice(
                    b * norb,
                    (b + 1) * norb,
                )

                S[rows, cols] = (
                    moments[a + b]
                )

                T[rows, cols] = (
                    moments[a + b + 1]
                )

        # Particularly useful for noisy quantum moments.
        S = hermitize(S)
        T = hermitize(T)

        # --------------------------------------------------
        # Orthogonalize Krylov metric
        # --------------------------------------------------

        svals, U = np.linalg.eigh(S)

        keep = (
            svals > threshold
        )

        if not np.any(keep):
            raise ValueError(
                "Block overlap matrix has no "
                "eigenvalues above threshold."
            )

        U_keep = U[:, keep]
        s_keep = svals[keep]

        # Rectangular inverse square root:
        #
        # X^dag S X = I
        #
        S_inv_sqrt = (
            U_keep
            @ np.diag(
                1.0 / np.sqrt(s_keep)
            )
        )

        # --------------------------------------------------
        # Effective auxiliary Hamiltonian
        # --------------------------------------------------

        F = (
            S_inv_sqrt.conj().T
            @ T
            @ S_inv_sqrt
        )

        F = hermitize(F)

        poles, C = np.linalg.eigh(F)

        # --------------------------------------------------
        # Coupling of physical orbital space to
        # reconstructed auxiliary eigenstates
        # --------------------------------------------------

        B = np.zeros(
            (dim, norb),
            dtype=complex,
        )

        for a in range(nb):

            rows = slice(
                a * norb,
                (a + 1) * norb,
            )

            B[rows, :] = moments[a]

        X = (
            B.conj().T
            @ S_inv_sqrt
            @ C
        )

        residues = []

        for k in range(len(poles)):

            x = X[:, k:k + 1]

            residues.append(
                x @ x.conj().T
            )

        if verbose:

            print(
                f"nblocks       = {nb}"
            )

            print(
                f"block dimension = {dim}"
            )

            print(
                "S eigenvalues =",
                svals,
            )

            print(
                "retained rank =",
                np.count_nonzero(keep),
            )

            print(
                "poles =",
                poles,
            )

        return {
            "poles": poles,
            "residues": residues,
            "couplings": X,
            "overlap_eigenvalues": svals,
            "retained_rank":
                np.count_nonzero(keep),
            "nblocks": nb,
        }

    # ==========================================================
    # Reconstruct hole and particle sectors
    # ==========================================================

    hole_aux = block_lancsoz(
        hole_moments
    )

    particle_aux = block_lancsoz(
        particle_moments
    )

    hole_poles = hole_aux[
        "poles"
    ]

    particle_poles = particle_aux[
        "poles"
    ]

    hole_residues = hole_aux[
        "residues"
    ]

    particle_residues = particle_aux[
        "residues"
    ]

    # ==========================================================
    # Green's function
    # ==========================================================

    nw = len(omega_grid)

    G_hole = np.zeros(
        (nw, norb, norb),
        dtype=complex,
    )

    G_particle = np.zeros_like(
        G_hole
    )

    for iw, omega in enumerate(
        omega_grid
    ):

        for pole, R in zip(
            hole_poles,
            hole_residues,
        ):

            G_hole[iw] += (
                R
                / (
                    omega
                    - pole
                    + 1j * eta
                )
            )

        for pole, R in zip(
            particle_poles,
            particle_residues,
        ):

            G_particle[iw] += (
                R
                / (
                    omega
                    - pole
                    + 1j * eta
                )
            )

    G = (
        G_hole
        + G_particle
    )

    # ==========================================================
    # Spectral functions
    # ==========================================================

    A_hole = (
        -1.0 / np.pi
        * np.imag(
            np.trace(
                G_hole,
                axis1=1,
                axis2=2,
            )
        )
    )

    A_particle = (
        -1.0 / np.pi
        * np.imag(
            np.trace(
                G_particle,
                axis1=1,
                axis2=2,
            )
        )
    )

    A = (
        A_hole
        + A_particle
    )

    # ==========================================================
    # Galitskii-Migdal energy
    # ==========================================================

    gm_energy = None

    if one_body_integrals is not None:

        h1 = np.asarray(
            one_body_integrals,
            dtype=complex,
        )

        if h1.shape != (
            norb,
            norb,
        ):
            raise ValueError(
                "one_body_integrals must have "
                "shape (norb, norb)."
            )

        if hole_moments.shape[0] < 2:
            raise ValueError(
                "At least moments n=0 and n=1 "
                "are required for the "
                "Galitskii-Migdal energy."
            )

        T0 = hole_moments[0]
        T1 = hole_moments[1]

        E_electronic = 0.5 * (
            np.einsum(
                "ij,ji->",
                T0,
                h1,
            )
            + np.trace(T1)
        ).real

        E_total = (
            E_electronic
            + constant_energy
        )

        gm_energy = {
            "electronic": E_electronic,
            "constant": constant_energy,
            "total": E_total,
        }

    # ==========================================================
    # Return everything
    # ==========================================================

    return {
        "omega": omega_grid,

        "G": G,
        "G_hole": G_hole,
        "G_particle": G_particle,

        "A": A,
        "A_hole": A_hole,
        "A_particle": A_particle,

        "hole_poles": hole_poles,
        "particle_poles":
            particle_poles,

        "hole_residues":
            hole_residues,
        "particle_residues":
            particle_residues,

        "hole_couplings":
            hole_aux["couplings"],
        "particle_couplings":
            particle_aux["couplings"],

        "hole_auxiliary":
            hole_aux,
        "particle_auxiliary":
            particle_aux,

        "galitskii_migdal":
            gm_energy,
    }