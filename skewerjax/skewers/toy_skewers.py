import numbers
import secrets

import jax
import jax.numpy as jnp
from jax.numpy.fft import rfftfreq, irfft


def _as_prng_key(rng=None, seed=None):
    """
    Convert rng/seed into a JAX PRNGKey.

    Accepted inputs
    ---------------
    rng : jax PRNGKey, int, or None
        If None, a key is built from `seed`.
    seed : int or None
        Used only when rng is None.

    Notes
    -----
    - If both rng and seed are None, a random Python-side seed is generated.
    - This preserves the old behavior of getting fresh randomness by default.
    """
    if rng is None:
        if seed is None:
            seed = secrets.randbits(32)
        return jax.random.PRNGKey(int(seed))

    if isinstance(rng, numbers.Integral):
        return jax.random.PRNGKey(int(rng))

    return rng


class ShiftedGaussianTauGenerator1D:
    """
    Generate n_fields optical-depth-like fields tau_i(v) from a single
    1D Gaussian random field tau_ref(v), using periodic shifts.

    Returns only:
        tau_fields with shape (N_samples, n_fields, Npix)

    Parameters
    ----------
    Npix : int
        Number of pixels.
    L_box : float
        Box/domain length.
    delta_v_list : array-like
        Shifts applied to the reference field.
    pk_func : callable
        Function of the form pk_func(k) returning the target 1D power spectrum.
    mean_tau : float, optional
        Mean added to the sampled GRF.
    contrast : float, optional
        Rescales the fluctuation amplitude after standardization.
    round_shifts_to_grid : bool, optional
        If True, shifts are rounded to the nearest grid pixel.
    """

    def __init__(
        self,
        Npix,
        L_box,
        delta_v_list,
        pk_func,
        mean_tau=1.0,
        contrast=0.35,
        round_shifts_to_grid=True,
    ):
        self.Npix = int(Npix)
        self.L_box = float(L_box)
        self.delta_v_list = jnp.asarray(delta_v_list, dtype=float)
        self.n_fields = len(self.delta_v_list)

        self.dv = self.L_box / self.Npix
        self.v = jnp.arange(self.Npix) * self.dv
        self.k = 2.0 * jnp.pi * rfftfreq(self.Npix, d=self.dv)

        self.mean_tau = float(mean_tau)
        self.contrast = float(contrast)
        self.round_shifts_to_grid = bool(round_shifts_to_grid)

        if pk_func is None:
            raise ValueError("You must provide a pk_func(k) callable.")
        self.pk_func = pk_func

        # pk_func is evaluated once here. This is fully JAX only if pk_func itself
        # is written with jax.numpy; otherwise it still works eagerly if it accepts
        # array-like input.
        self.Pk_target = jnp.asarray(self.pk_func(self.k), dtype=float)

        if self.Pk_target.shape != self.k.shape:
            raise ValueError("pk_func(k) must return an array with the same shape as k.")

        if bool(jnp.any(self.Pk_target < 0)):
            raise ValueError("pk_func(k) must return a non-negative power spectrum.")

        if self.round_shifts_to_grid:
            self.shift_pix = jnp.rint(self.delta_v_list / self.dv).astype(jnp.int32)
        else:
            shift_exact = self.delta_v_list / self.dv
            if not bool(jnp.allclose(shift_exact, jnp.round(shift_exact))):
                raise ValueError(
                    "delta_v_list must be integer multiples of dv when "
                    "round_shifts_to_grid=False."
                )
            self.shift_pix = shift_exact.astype(jnp.int32)

    def _sample_tau_ref(self, N_samples, rng):
        N_samples = int(N_samples)
        nfreq = self.k.size

        sigma_k = jnp.sqrt((self.Npix / self.dv) * self.Pk_target / 2.0)

        # Match precision to the real dtype
        complex_dtype = (self.Pk_target + 0j).dtype
        fk = jnp.zeros((N_samples, nfreq), dtype=complex_dtype)

        key_re, key_im, key_nyq = jax.random.split(rng, 3)

        if nfreq > 2:
            re = jax.random.normal(
                key_re,
                shape=(N_samples, nfreq - 2),
                dtype=self.Pk_target.dtype,
            )
            im = jax.random.normal(
                key_im,
                shape=(N_samples, nfreq - 2),
                dtype=self.Pk_target.dtype,
            )

            fk = fk.at[:, 1:-1].set((re + 1j * im) * sigma_k[None, 1:-1])

        if self.Npix % 2 == 0:
            nyq = jax.random.normal(
                key_nyq,
                shape=(N_samples,),
                dtype=self.Pk_target.dtype,
            )
            nyq = nyq * jnp.sqrt((self.Npix / self.dv) * self.Pk_target[-1])
            fk = fk.at[:, -1].set(nyq)

        tau_fluct = irfft(fk, n=self.Npix, axis=1)

        std = jnp.std(tau_fluct, axis=1, keepdims=True)
        std = jnp.where(std == 0.0, 1.0, std)

        tau_ref = self.mean_tau + self.contrast * tau_fluct / std
        return tau_ref

    def sample(self, N_samples, rng=None, seed=None):
        """
        Generate tau_fields of shape (N_samples, n_fields, Npix).

        Parameters
        ----------
        N_samples : int
            Number of realizations.
        rng : jax PRNGKey, int, or None
            Random key. If None, a key is created from `seed`.
        seed : int or None
            Used only when rng is None.

        Returns
        -------
        tau_fields : jax.Array
            Shape (N_samples, n_fields, Npix)
        """
        rng = _as_prng_key(rng, seed)

        tau_ref = self._sample_tau_ref(int(N_samples), rng)

        tau_fields = jax.vmap(
            lambda shift: jnp.roll(tau_ref, -shift, axis=1),
            in_axes=0,
            out_axes=1,
        )(self.shift_pix)

        return tau_fields


class ShiftedSpikeTauGenerator1D:
    """
    Generate n_fields optical-depth-like fields tau_i(v) from a common set
    of randomly sampled Gaussian spikes, using periodic shifts.

    Each field i is built as
        tau_i(v) = A_i * sum_p exp[-0.5 * (d_i / sigma_i)^2]
    where the spike centers are shifted by delta_v_i with periodic wrapping.

    Returns only:
        tau_fields with shape (N_samples, n_fields, Npix)
    """

    def __init__(
        self,
        Npix,
        L_box,
        delta_v_list,
        sigma_list,
        amplitude_list,
        n_pairs,
    ):
        self.Npix = int(Npix)
        self.L_box = float(L_box)
        self.n_pairs = int(n_pairs)

        self.dv = self.L_box / self.Npix
        self.v = jnp.arange(self.Npix) * self.dv

        self.delta_v_list = jnp.asarray(delta_v_list, dtype=float)
        self.n_fields = len(self.delta_v_list)

        self.sigma_list = jnp.asarray(sigma_list, dtype=float)
        self.amplitude_list = jnp.asarray(amplitude_list, dtype=float)

        if self.sigma_list.ndim == 0:
            self.sigma_list = jnp.full((self.n_fields,), self.sigma_list)

        if self.amplitude_list.ndim == 0:
            self.amplitude_list = jnp.full((self.n_fields,), self.amplitude_list)

        if len(self.sigma_list) != self.n_fields:
            raise ValueError("sigma_list must have length n_fields or be a scalar.")
        if len(self.amplitude_list) != self.n_fields:
            raise ValueError("amplitude_list must have length n_fields or be a scalar.")

    def sample(self, N_samples, rng=None, seed=None):
        """
        Generate tau_fields of shape (N_samples, n_fields, Npix).

        Parameters
        ----------
        N_samples : int
            Number of realizations.
        rng : jax PRNGKey, int, or None
            Random key. If None, a key is created from `seed`.
        seed : int or None
            Used only when rng is None.

        Returns
        -------
        tau_fields : jax.Array
            Shape (N_samples, n_fields, Npix)
        """
        rng = _as_prng_key(rng, seed)
        N_samples = int(N_samples)

        # Base random centers shared by all fields
        centers_ref = jax.random.uniform(
            rng,
            shape=(N_samples, self.n_pairs),
            minval=0.0,
            maxval=self.L_box,
        )

        # Shifted centers for each field
        centers_fields = (
            centers_ref[:, None, :] + self.delta_v_list[None, :, None]
        ) % self.L_box
        # shape: (N_samples, n_fields, n_pairs)

        # Periodic distances to every pixel
        dist = (
            (self.v[None, None, None, :] - centers_fields[:, :, :, None] + self.L_box / 2)
            % self.L_box
        ) - self.L_box / 2
        # shape: (N_samples, n_fields, n_pairs, Npix)

        # Gaussian spikes with per-field sigma
        spikes = jnp.exp(
            -0.5 * (dist / self.sigma_list[None, :, None, None]) ** 2
        )

        # Sum over pairs and multiply by per-field amplitude
        tau_fields = (
            self.amplitude_list[None, :, None] * jnp.sum(spikes, axis=2)
        )
        # shape: (N_samples, n_fields, Npix)

        return tau_fields