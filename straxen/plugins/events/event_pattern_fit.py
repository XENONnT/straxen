import strax
import straxen
import numpy as np
import numba
from straxen.numbafied_scipy import numba_gammaln, numba_betainc
from scipy.special import loggamma

export, __all__ = strax.exporter()


@export
class EventPatternFit(strax.Plugin):
    """Plugin that provides patter information for events."""

    depends_on = ("event_area_per_channel", "event_basics", "event_positions")
    save_when = strax.SaveWhen.ALWAYS
    provides = "event_pattern_fit"
    __version__ = "0.1.3"

    # Getting S1 AFT maps
    s1_aft_map = straxen.URLConfig(
        default=(
            "itp_map://resource://cmt://s1_aft_xyz_map?version=ONLINE&run_id=plugin.run_id&fmt=json"
        ),
        cache=True,
    )

    electron_drift_velocity = straxen.URLConfig(
        default="cmt://electron_drift_velocity?version=ONLINE&run_id=plugin.run_id",
        cache=True,
        help="Vertical electron drift velocity in cm/ns (1e4 m/ms)",
    )

    electron_drift_time_gate = straxen.URLConfig(
        default="cmt://electron_drift_time_gate?version=ONLINE&run_id=plugin.run_id",
        help="Electron drift time from the gate in ns",
        cache=True,
    )

    s1_optical_map = straxen.URLConfig(
        help="S1 (x, y, z) optical/pattern map.",
        infer_type=False,
        default=(
            "itp_map://"
            "resource://"
            "XENONnT_s1_xyz_patterns_corrected_qes_MCva43fa9b_wires.pkl"
            "?fmt=pkl"
        ),
    )

    s2_optical_map = straxen.URLConfig(
        help="S2 (x, y) optical/pattern map.",
        infer_type=False,
        default=(
            "itp_map://"
            "resource://"
            "XENONnT_s2_xy_patterns_LCE_corrected_qes_MCva43fa9b_wires.pkl"
            "?fmt=pkl"
        ),
    )

    s2_tf_model = straxen.URLConfig(
        help="S2 (x, y) optical data-driven model",
        infer_type=False,
        default=(
            "tf://"
            "resource://"
            "XENONnT_s2_optical_map_data_driven_ML_v0_2021_11_25.tar.gz"
            "?custom_objects=plugin.s2_map_custom_objects"
            "&fmt=abs_path"
        ),
    )

    mean_pe_per_photon = straxen.URLConfig(
        help="Mean of full VUV single photon response",
        default=1.2,
        infer_type=False,
    )

    gain_model = straxen.URLConfig(
        infer_type=False, help="PMT gain model. Specify as (model_type, model_config)"
    )

    n_tpc_pmts = straxen.URLConfig(type=int, help="Number of TPC PMTs")

    n_top_pmts = straxen.URLConfig(type=int, help="Number of top TPC PMTs")

    s1_min_area_pattern_fit = straxen.URLConfig(
        infer_type=False,
        help="Skip EventPatternFit reconstruction if S1 area (PE) is less than this",
        default=2,
    )

    s2_min_area_pattern_fit = straxen.URLConfig(
        infer_type=False,
        help="Skip EventPatternFit reconstruction if S2 area (PE) is less than this",
        default=10,
    )

    store_per_channel = straxen.URLConfig(
        default=False, type=bool, help="Store normalized LLH per channel for each peak"
    )

    max_r_pattern_fit = straxen.URLConfig(
        default=straxen.tpc_r,
        type=float,
        help="Maximal radius of the peaks where llh calculation will be performed",
    )

    def infer_dtype(self):
        dtype = [
            ("s2_2llh", np.float32, "Modified Poisson likelihood value for main S2 in the event"),
            (
                "s2_neural_2llh",
                np.float32,
                "Data-driven based likelihood value for main S2 in the event",
            ),
            ("alt_s2_2llh", np.float32, "Modified Poisson likelihood value for alternative S2"),
            (
                "alt_s2_neural_2llh",
                np.float32,
                "Data-driven based likelihood value for alternative S2 in the event",
            ),
            ("s1_2llh", np.float32, "Modified Poisson likelihood value for main S1"),
            (
                "s1_top_2llh",
                np.float32,
                "Modified Poisson likelihood value for main S1, calculated from top array",
            ),
            (
                "s1_bottom_2llh",
                np.float32,
                "Modified Poisson likelihood value for main S1, calculated from bottom array",
            ),
            (
                "s1_area_fraction_top_continuous_probability",
                np.float32,
                "Continuous binomial test for S1 area fraction top",
            ),
            (
                "s1_area_fraction_top_discrete_probability",
                np.float32,
                "Discrete binomial test for S1 area fraction top",
            ),
            (
                "s1_photon_fraction_top_continuous_probability",
                np.float32,
                "Continuous binomial test for S1 photon fraction top",
            ),
            (
                "s1_photon_fraction_top_discrete_probability",
                np.float32,
                "Discrete binomial test for S1 photon fraction top",
            ),
            (
                "alt_s1_area_fraction_top_continuous_probability",
                np.float32,
                "Continuous binomial test for alternative S1 area fraction top",
            ),
            (
                "alt_s1_area_fraction_top_discrete_probability",
                np.float32,
                "Discrete binomial test for alternative S1 area fraction top",
            ),
            (
                "alt_s1_photon_fraction_top_continuous_probability",
                np.float32,
                "Continuous binomial test for alternative S1 photon fraction top",
            ),
            (
                "alt_s1_photon_fraction_top_discrete_probability",
                np.float32,
                "Discrete binomial test for alternative S1 photon fraction top",
            ),
        ]

        if self.store_per_channel:
            dtype += [
                (
                    ("2LLH per channel for main S2", "s2_2llh_per_channel"),
                    np.float32,
                    (self.n_top_pmts,),
                ),
                (
                    ("2LLH per channel for alternative S2", "alt_s2_2llh_per_channel"),
                    np.float32,
                    (self.n_top_pmts,),
                ),
                (("Pattern main S2", "s2_pattern"), np.float32, (self.n_top_pmts,)),
                (("Pattern alt S2", "alt_s2_pattern"), np.float32, (self.n_top_pmts,)),
                (("Pattern for main S1", "s1_pattern"), np.float32, (self.n_tpc_pmts,)),
                (
                    ("2LLH per channel for main S1", "s1_2llh_per_channel"),
                    np.float32,
                    (self.n_tpc_pmts,),
                ),
            ]
        dtype += strax.time_fields
        return dtype

    @property
    def s2_map_custom_objects(self):
        def _logl_loss(patterns_true, likelihood):
            return likelihood / 10.0

        return {"_logl_loss": _logl_loss}

    def setup(self):
        # FIXME: Consider renaming the configs to match usage

        self.to_pe = self.gain_model

        self.mean_pe_photon = self.mean_pe_per_photon

        # Getting optical maps
        self.s1_pattern_map = self.s1_optical_map
        self.s2_pattern_map = self.s2_optical_map

        # Getting S2 data-driven tensorflow models
        self.model = self.s2_tf_model

        import tensorflow as tf

        self.model_chi2 = tf.keras.Model(
            self.model.inputs, self.model.get_layer("Likelihood").output
        )

        # Getting gain model to get dead PMTs
        self.dead_PMTs = np.where(self.to_pe == 0)[0]
        self.pmtbool = ~np.in1d(np.arange(0, self.n_tpc_pmts), self.dead_PMTs)
        self.pmtbool_top = self.pmtbool[: self.n_top_pmts]
        self.pmtbool_bottom = self.pmtbool[self.n_top_pmts : self.n_tpc_pmts]

    def compute(self, events):
        result = np.zeros(len(events), dtype=self.dtype)
        result["time"] = events["time"]
        result["endtime"] = strax.endtime(events)

        # Computing LLH values for S1s
        self.compute_s1_llhvalue(events, result)

        # Computing LLH values for S2s
        self.compute_s2_llhvalue(events, result)

        # Computing chi2 values for S2s
        self.compute_s2_neural_llhvalue(events, result)

        # Computing binomial test for s1 area fraction top
        positions = np.vstack([events["x"], events["y"], events["z"]]).T
        aft_prob = self.s1_aft_map(positions)

        alt_s1_interaction_drift_time = events["s2_center_time"] - events["alt_s1_center_time"]
        alt_s1_interaction_z = -self.electron_drift_velocity * (
            alt_s1_interaction_drift_time - self.electron_drift_time_gate
        )
        alt_positions = np.vstack([events["x"], events["y"], alt_s1_interaction_z]).T
        alt_aft_prob = self.s1_aft_map(alt_positions)

        # main s1 events
        mask_s1 = ~np.isnan(aft_prob)
        mask_s1 &= ~np.isnan(events["s1_area"])
        mask_s1 &= ~np.isnan(events["s1_area_fraction_top"])

        # default value is nan, it will be overwrite if the event satisfy the requirements
        result["s1_area_fraction_top_continuous_probability"][:] = np.nan
        result["s1_area_fraction_top_discrete_probability"][:] = np.nan
        result["s1_photon_fraction_top_continuous_probability"][:] = np.nan
        result["s1_photon_fraction_top_discrete_probability"][:] = np.nan

        # compute binomial test only if we have events that have valid aft prob, s1 area and s1 aft
        if np.sum(mask_s1):
            arg = (
                aft_prob[mask_s1],
                events["s1_area"][mask_s1],
                events["s1_area_fraction_top"][mask_s1],
            )
            result["s1_area_fraction_top_continuous_probability"][mask_s1] = (
                s1_area_fraction_top_probability(*arg)
            )
            result["s1_area_fraction_top_discrete_probability"][mask_s1] = (
                s1_area_fraction_top_probability(*arg, "discrete")
            )
            arg = (
                aft_prob[mask_s1],
                events["s1_area"][mask_s1] / self.mean_pe_per_photon,
                events["s1_area_fraction_top"][mask_s1],
            )
            result["s1_photon_fraction_top_continuous_probability"][mask_s1] = (
                s1_area_fraction_top_probability(*arg)
            )
            result["s1_photon_fraction_top_discrete_probability"][mask_s1] = (
                s1_area_fraction_top_probability(*arg, "discrete")
            )

        # alternative s1 events
        mask_alt_s1 = ~np.isnan(alt_aft_prob)
        mask_alt_s1 &= ~np.isnan(events["alt_s1_area"])
        mask_alt_s1 &= ~np.isnan(events["alt_s1_area_fraction_top"])

        # default value is nan, it will be ovewrite if the event satisfy the requirments
        result["alt_s1_area_fraction_top_continuous_probability"][:] = np.nan
        result["alt_s1_area_fraction_top_discrete_probability"][:] = np.nan
        result["alt_s1_photon_fraction_top_continuous_probability"][:] = np.nan
        result["alt_s1_photon_fraction_top_discrete_probability"][:] = np.nan

        # compute binomial test only if we have events that have valid aft prob,
        # alt s1 area and alt s1 aft
        if np.sum(mask_alt_s1):
            arg = (
                alt_aft_prob[mask_alt_s1],
                events["alt_s1_area"][mask_alt_s1],
                events["alt_s1_area_fraction_top"][mask_alt_s1],
            )
            result["alt_s1_area_fraction_top_continuous_probability"][mask_alt_s1] = (
                s1_area_fraction_top_probability(*arg)
            )
            result["alt_s1_area_fraction_top_discrete_probability"][mask_alt_s1] = (
                s1_area_fraction_top_probability(*arg, "discrete")
            )
            arg = (
                alt_aft_prob[mask_alt_s1],
                events["alt_s1_area"][mask_alt_s1] / self.mean_pe_per_photon,
                events["alt_s1_area_fraction_top"][mask_alt_s1],
            )
            result["alt_s1_photon_fraction_top_continuous_probability"][mask_alt_s1] = (
                s1_area_fraction_top_probability(*arg)
            )
            result["alt_s1_photon_fraction_top_discrete_probability"][mask_alt_s1] = (
                s1_area_fraction_top_probability(*arg, "discrete")
            )

        return result

    def compute_s1_llhvalue(self, events, result):
        # Selecting S1s for pattern fit calculation
        # - must exist (index != -1)
        # - must have total area larger minimal one
        # - must have positive AFT
        x, y, z = events["x"], events["y"], events["z"]
        cur_s1_bool = events["s1_area"] > self.s1_min_area_pattern_fit
        cur_s1_bool &= events["s1_index"] != -1
        cur_s1_bool &= events["s1_area_fraction_top"] >= 0
        cur_s1_bool &= np.isfinite(x)
        cur_s1_bool &= np.isfinite(y)
        cur_s1_bool &= np.isfinite(z)
        cur_s1_bool &= (x**2 + y**2) < self.max_r_pattern_fit**2

        # default value is nan, it will be ovewrite if the event satisfy the requirments
        result["s1_2llh"][:] = np.nan
        result["s1_top_2llh"][:] = np.nan
        result["s1_bottom_2llh"][:] = np.nan

        # Making expectation patterns [ in PE ]
        if np.sum(cur_s1_bool):
            s1_map_effs = self.s1_pattern_map(np.array([x, y, z]).T)[cur_s1_bool, :]
            s1_area = events["s1_area"][cur_s1_bool]
            s1_pattern = (
                s1_area[:, None]
                * (s1_map_effs[:, self.pmtbool])
                / np.sum(s1_map_effs[:, self.pmtbool], axis=1)[:, None]
            )

            s1_pattern_top = events["s1_area_fraction_top"][cur_s1_bool] * s1_area
            s1_pattern_top = s1_pattern_top[:, None] * (
                (s1_map_effs[:, : self.n_top_pmts])[:, self.pmtbool_top]
            )
            s1_pattern_top /= np.sum(
                (s1_map_effs[:, : self.n_top_pmts])[:, self.pmtbool_top], axis=1
            )[:, None]
            s1_pattern_bottom = (1 - events["s1_area_fraction_top"][cur_s1_bool]) * s1_area
            s1_pattern_bottom = s1_pattern_bottom[:, None] * (
                (s1_map_effs[:, self.n_top_pmts :])[:, self.pmtbool_bottom]
            )
            s1_pattern_bottom /= np.sum(
                (s1_map_effs[:, self.n_top_pmts :])[:, self.pmtbool_bottom], axis=1
            )[:, None]

            # Getting pattern from data
            s1_area_per_channel_ = events["s1_area_per_channel"][cur_s1_bool, :]
            s1_area_per_channel = s1_area_per_channel_[:, self.pmtbool]
            s1_area_per_channel_top = (s1_area_per_channel_[:, : self.n_top_pmts])[
                :, self.pmtbool_top
            ]
            s1_area_per_channel_bottom = (s1_area_per_channel_[:, self.n_top_pmts :])[
                :, self.pmtbool_bottom
            ]

            # Top and bottom
            arg1 = s1_pattern / self.mean_pe_photon, s1_area_per_channel, self.mean_pe_photon
            arg2 = (
                s1_area_per_channel / self.mean_pe_photon,
                s1_area_per_channel,
                self.mean_pe_photon,
            )
            norm_llh_val = neg2llh_modpoisson(*arg1) - neg2llh_modpoisson(*arg2)
            result["s1_2llh"][cur_s1_bool] = np.sum(norm_llh_val, axis=1)

            # If needed to stire - store only top and bottom array, but not together
            if self.store_per_channel:
                # Storring pattern information
                store_patterns = np.zeros((s1_pattern.shape[0], self.n_tpc_pmts))
                store_patterns[:, self.pmtbool] = s1_pattern
                result["s1_pattern"][cur_s1_bool] = store_patterns
                # Storing actual LLH values
                store_2LLH_ch = np.zeros((norm_llh_val.shape[0], self.n_tpc_pmts))
                store_2LLH_ch[:, self.pmtbool] = norm_llh_val
                result["s1_2llh_per_channel"][cur_s1_bool] = store_2LLH_ch

            # Top
            arg1 = (
                s1_pattern_top / self.mean_pe_photon,
                s1_area_per_channel_top,
                self.mean_pe_photon,
            )
            arg2 = (
                s1_area_per_channel_top / self.mean_pe_photon,
                s1_area_per_channel_top,
                self.mean_pe_photon,
            )
            norm_llh_val = neg2llh_modpoisson(*arg1) - neg2llh_modpoisson(*arg2)
            result["s1_top_2llh"][cur_s1_bool] = np.sum(norm_llh_val, axis=1)

            # Bottom
            arg1 = (
                s1_pattern_bottom / self.mean_pe_photon,
                s1_area_per_channel_bottom,
                self.mean_pe_photon,
            )
            arg2 = (
                s1_area_per_channel_bottom / self.mean_pe_photon,
                s1_area_per_channel_bottom,
                self.mean_pe_photon,
            )
            norm_llh_val = neg2llh_modpoisson(*arg1) - neg2llh_modpoisson(*arg2)
            result["s1_bottom_2llh"][cur_s1_bool] = np.sum(norm_llh_val, axis=1)

    def compute_s2_llhvalue(self, events, result):
        for t_ in ["s2", "alt_s2"]:
            # Selecting S2s for pattern fit calculation
            # - must exist (index != -1)
            # - must have total area larger minimal one
            # - must have positive AFT
            x, y = events[t_ + "_x"], events[t_ + "_y"]
            s2_mask = events[t_ + "_area"] > self.s2_min_area_pattern_fit
            s2_mask &= events[t_ + "_area_fraction_top"] > 0
            s2_mask &= (x**2 + y**2) < self.max_r_pattern_fit**2

            # default value is nan, it will be ovewrite if the event satisfy the requirments
            result[t_ + "_2llh"][:] = np.nan

            # Making expectation patterns [ in PE ]
            if np.sum(s2_mask):
                s2_map_effs = self.s2_pattern_map(np.array([x, y]).T)[s2_mask, 0 : self.n_top_pmts]
                s2_map_effs = s2_map_effs[:, self.pmtbool_top]
                s2_top_area = (events[t_ + "_area_fraction_top"] * events[t_ + "_area"])[s2_mask]
                s2_pattern = (
                    s2_top_area[:, None] * s2_map_effs / np.sum(s2_map_effs, axis=1)[:, None]
                )

                # Getting pattern from data
                s2_top_area_per_channel = events[t_ + "_area_per_channel"][
                    s2_mask, 0 : self.n_top_pmts
                ]
                s2_top_area_per_channel = s2_top_area_per_channel[:, self.pmtbool_top]

                # Calculating LLH, this is shifted Poisson
                # we get area expectation and we need to scale them to get
                # photon expectation
                norm_llh_val = neg2llh_modpoisson(
                    mu=s2_pattern / self.mean_pe_photon,
                    areas=s2_top_area_per_channel,
                    mean_pe_photon=self.mean_pe_photon,
                ) - neg2llh_modpoisson(
                    mu=s2_top_area_per_channel / self.mean_pe_photon,
                    areas=s2_top_area_per_channel,
                    mean_pe_photon=self.mean_pe_photon,
                )
                result[t_ + "_2llh"][s2_mask] = np.sum(norm_llh_val, axis=1)

                if self.store_per_channel:
                    store_patterns = np.zeros((s2_pattern.shape[0], self.n_top_pmts))
                    store_patterns[:, self.pmtbool_top] = s2_pattern
                    result[t_ + "_pattern"][s2_mask] = store_patterns

                    store_2LLH_ch = np.zeros((norm_llh_val.shape[0], self.n_top_pmts))
                    store_2LLH_ch[:, self.pmtbool_top] = norm_llh_val
                    result[t_ + "_2llh_per_channel"][s2_mask] = store_2LLH_ch

    def compute_s2_neural_llhvalue(self, events, result):
        for t_ in ["s2", "alt_s2"]:
            x, y = events[t_ + "_x"], events[t_ + "_y"]
            s2_mask = events[t_ + "_area"] > self.s2_min_area_pattern_fit
            s2_mask &= events[t_ + "_area_fraction_top"] > 0

            # default value is nan, it will be ovewrite if the event satisfy the requirements
            result[t_ + "_neural_2llh"][:] = np.nan

            # Produce position and top pattern to feed tensorflow model, return chi2/N
            if np.sum(s2_mask):
                s2_pos = np.stack((x, y)).T[s2_mask]
                s2_pat = events[t_ + "_area_per_channel"][s2_mask, 0 : self.n_top_pmts]
                # Output[0]: loss function, -2*log-likelihood, Output[1]: chi2
                result[t_ + "_neural_2llh"][s2_mask] = self.model_chi2.predict(
                    {"xx": s2_pos, "yy": s2_pat}, verbose=0
                )[1]


def neg2llh_modpoisson(mu=None, areas=None, mean_pe_photon=1.0):
    """Modified poisson distribution with proper normalization for shifted poisson.

    mu - expected number of photons per channel
    areas - observed areas per channel
    mean_pe_photon - mean of area responce for one photon

    """
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = areas / mean_pe_photon
        res = 2.0 * (
            mu - (fraction) * np.log(mu) + loggamma((fraction) + 1) + np.log(mean_pe_photon)
        )
    is_zero = areas <= 0  # If area equals or smaller than 0 - assume 0
    res[is_zero] = 2.0 * mu[is_zero]
    # if zero channel has negative expectation, assume LLH to be 0 there
    # this happens in the normalization factor calculation when mu is received from area
    neg_mu = mu < 0.0
    res[is_zero | neg_mu] = 0.0
    return res


# continuous and discrete binomial test


@numba.njit
def lbinom_pmf(k, n, p):
    """Log of binomial probability mass function approximated with gamma function."""
    scale_log = numba_gammaln(n + 1) - numba_gammaln(n - k + 1) - numba_gammaln(k + 1)
    ret_log = scale_log + k * np.log(p) + (n - k) * np.log(1 - p)
    return ret_log


@numba.njit
def binom_pmf(k, n, p):
    """Binomial probability mass function approximated with gamma function."""
    return np.exp(lbinom_pmf(k, n, p))


@numba.njit
def binom_cdf(k, n, p):
    if k >= n:
        return 1.0
    return numba_betainc(n - k, k + 1, 1.0 - p)


@numba.njit
def binom_sf(k, n, p):
    return 1 - binom_cdf(k, n, p)


@numba.njit
def lbinom_pmf_diriv(k, n, p, dk=1e-7):
    """Numerical dirivitive of Binomial pmf approximated with gamma function."""
    if k + dk < n:
        return (lbinom_pmf(k + dk, n, p) - lbinom_pmf(k, n, p)) / dk
    else:
        return (lbinom_pmf(k - dk, n, p) - lbinom_pmf(k, n, p)) / -dk


@numba.njit(cache=True)
def _numeric_derivative(y0, y1, err, target, x_min, x_max, x0, x1):
    """Get close to <target> by doing a numeric derivative."""
    if abs(y1 - y0) < err:
        # break by passing dx == 0
        if abs(y0 - target) < abs(y1 - target):
            x = x0
        else:
            x = x1
        return 0.0, x, x

    x = (target - y0) / (y1 - y0) * (x1 - x0) + x0
    x = min(x, x_max)
    x = max(x, x_min)

    dx = abs(x - x1)
    x0 = x1
    x1 = x

    return dx, x0, x1


@numba.njit
def lbinom_pmf_mode(x_min, x_max, target, args, err=1e-7, max_iter=50):
    """Find the root of the derivative of log Binomial pmf with secant method."""
    x0 = x_min
    x1 = x_max
    dx = abs(x1 - x0)

    while (dx > err) and (max_iter > 0):
        y0 = lbinom_pmf_diriv(x0, *args)
        y1 = lbinom_pmf_diriv(x1, *args)
        dx, x0, x1 = _numeric_derivative(y0, y1, err, target, x_min, x_max, x0, x1)
        max_iter -= 1
    return x1


@numba.njit
def lbinom_pmf_inverse(x_min, x_max, target, args, err=1e-7, max_iter=50):
    """Find the where the log Binomial pmf cross target with secant method."""
    x0 = x_min
    x1 = x_max
    dx = abs(x1 - x0)

    if dx != 0:
        while (dx > err) and (max_iter > 0):
            y0 = lbinom_pmf(x0, *args)
            y1 = lbinom_pmf(x1, *args)
            dx, x0, x1 = _numeric_derivative(y0, y1, err, target, x_min, x_max, x0, x1)
            max_iter -= 1
        if x0 == x1 == 0 and y0 - target > 0:
            x1 = np.nan
        if x0 == x1 == args[0] and y0 - target < 0:
            x1 = np.nan
    else:
        x1 = np.nan

    return x1


@numba.njit
def binom_test(k, n, p):
    """The main purpose of this algorithm is to find the value j on the other side of the mode that
    has the same probability as k, and integrate the tails outward from k and j.

    In the case where either k or j are zero, only the non-zero tail is integrated.

    """
    mode = lbinom_pmf_mode(0, n, 0, (n, p))
    distance = abs(mode - k)
    target = lbinom_pmf(k, n, p)

    if k < mode:
        j_min = mode
        j_max = min(mode + 2.0 * distance, n)
        j = lbinom_pmf_inverse(j_min, j_max, target, (n, p))
        ls, rs = k, j
    elif k > mode:
        j_min = max(mode - 2.0 * distance, 0)
        j_max = mode
        j = lbinom_pmf_inverse(j_min, j_max, target, (n, p))
        ls, rs = j, k
    else:
        return 1

    pval = 0
    if not np.isnan(ls):
        pval += binom_cdf(ls, n, p)
    if not np.isnan(rs):
        pval += binom_sf(rs, n, p)
        if np.isnan(ls):
            pval += binom_pmf(rs, n, p)

    return pval


@np.vectorize
@numba.njit
def s1_area_fraction_top_probability(aft_prob, area_tot, area_fraction_top, mode="continuous"):
    """Function to compute the S1 AFT probability."""
    area_top = area_tot * area_fraction_top

    # Raise a warning in case one of these three condition is verified
    # and return binomial test equal to nan since they are not physical
    # k: size_top, n: size_tot, p: aft_prob
    do_test = True
    if area_tot < area_top:
        # warnings.warn(f'n {area_tot} must be >= k {area_top}')
        binomial_test = np.nan
        do_test = False
    if (aft_prob > 1.0) or (aft_prob < 0.0):
        # warnings.warn(f'p {aft_prob} must be in range [0, 1]')
        binomial_test = np.nan
        do_test = False
    if area_top < 0:
        # warnings.warn(f'k {area_top} must be >= 0')
        binomial_test = np.nan
        do_test = False

    if do_test:
        if mode == "discrete":
            binomial_test = binom_pmf(area_top, area_tot, aft_prob)
            # TODO:
            # binomial_test = binomtest(
            #     k=round(area_top), n=round(area_tot), p=aft_prob, alternative="two-sided"
            # ).pvalue
        else:
            binomial_test = binom_test(area_top, area_tot, aft_prob)

    return binomial_test
