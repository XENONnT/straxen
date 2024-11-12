import numpy as np
import strax
import straxen
from straxen.plugins.defaults import DEFAULT_POSREC_ALGO


export, __all__ = strax.exporter()


@export
class EventPositionContour(strax.Plugin):
    """A strax plugin that computes event position contours and applies field distortion corrections
    using the conditional normalising flow model. For information on the model, see note_.

    .. _note: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:juehang:flow_posrec_proposal_sr2 # noqa: E501

    This plugin calculates position contours for S2 and alternative S2 signals, and applies
    field distortion corrections to improve the accuracy of position reconstruction.

    Attributes:
        __version__ (str): Version of the plugin.
        depends_on (tuple): Input data types required for this plugin.
        provides (str): Output data type provided by this plugin.
        compressor (str): Compression algorithm used for data storage.
        data_kind (str): Kind of data this plugin processes.
        loop_over (str): Data structure to iterate over during computation.

    Configuration options:
        n_poly (int): Number of points in the uncertainty contour.
        electron_drift_velocity (float): Electron drift velocity in cm/ns.
        fdc_map (str): Path to the 3D field distortion correction map.
        use_fdc_for_contour (bool): Whether to use full FDC for
        position reconstruction uncertainty contours.

    """

    __version__ = "0.1.0"

    depends_on = ("event_basics", "event_positions", "peak_positions_cnf", "peak_basics")

    provides = "event_position_contour"
    compressor = "zstd"
    data_kind = "events"
    loop_over = "events"

    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO, help="default reconstruction algorithm that provides (x,y)"
    )

    n_poly = straxen.URLConfig(
        default=16,
        infer_type=False,
        help="Size of uncertainty contour",
    )

    electron_drift_velocity = straxen.URLConfig(
        default="cmt://electron_drift_velocity?version=ONLINE&run_id=plugin.run_id",
        cache=True,
        help="Vertical electron drift velocity in cm/ns (1e4 m/ms)",
    )

    fdc_map = straxen.URLConfig(
        infer_type=False,
        help="3D field distortion correction map path",
        default="legacy-fdc://xenon1t_sr0_sr1?run_id=plugin.run_id",
    )

    use_fdc_for_contour = straxen.URLConfig(
        infer_type=False,
        help=(
            "Whether to use full FDC for position reconstruction uncertainty contours. "
            "Uses the FDC of the central point if false."
        ),
        default=False,
    )

    def infer_dtype(self):
        """Infer the data type for the output array.

        Returns:
            list: A list of tuples defining the structure of the output array.

        """
        # Define information lines for different signal types
        infoline = {
            "s1": "main S1",
            "s2": "main S2",
            "alt_s1": "alternative S1",
            "alt_s2": "alternative S2",
        }
        dtype = []

        # Add fields for position contours
        ptypes = ["s2", "alt_s2"]
        for type_ in ptypes:
            dtype += [
                (
                    (
                        f"Naive flow position contour for {infoline[type_]}",
                        f"{type_}_position_contour_cnf_naive",
                    ),
                    np.float32,
                    (self.n_poly + 1, 2),
                )
            ]

        # Add fields for flow position contour and corrected positions
        dtype += [
            (
                (f"Flow position contour", "position_contour_cnf"),
                np.float32,
                (self.n_poly + 1, 2),
            ),
            (
                (f"Flow x position", "x_cnf_fdc"),
                np.float32,
            ),
            (
                (f"Flow y position", "y_cnf_fdc"),
                np.float32,
            ),
        ]
        dtype += strax.time_fields
        return dtype

    def setup(self):
        """Set up the plugin by initializing coordinate scales and loading the FDC map."""
        self.coordinate_scales = [1.0, 1.0, -self.electron_drift_velocity]
        self.map = self.fdc_map

    def compute(self, events, peaks):
        """Compute event position contours and apply field distortion corrections.

        Args:
            events (np.ndarray): Input events array.
            peaks (np.ndarray): Input peaks array.

        Returns:
            np.ndarray: Array containing computed position contours and corrected positions.

        """
        # Initialize the result array
        result = np.zeros(len(events), dtype=self.dtype)
        result["time"] = events["time"]
        result["endtime"] = strax.endtime(events)

        # Initialize contour fields with NaN values
        for type_ in ["s2", "alt_s2"]:
            result[f"{type_}_position_contour_cnf_naive"] *= np.nan
        result[f"position_contour_cnf"] *= np.nan

        # Split peaks by containment in events
        split_peaks = strax.split_by_containment(peaks, events)

        # Process each event
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            for type_ in ["s2", "alt_s2"]:
                type_index = event[f"{type_}_index"]
                if type_index != -1:
                    # Store naive flow position contour
                    result[f"{type_}_position_contour_cnf_naive"][event_i] = sp[
                        "position_contours_cnf"
                    ][type_index]

                if type_ == "s2":
                    if self.use_fdc_for_contour:
                        # Apply full field distortion correction to contour
                        contour_plus_xy = np.concatenate(
                            [
                                sp["position_contours_cnf"][type_index],
                                np.array([[sp["x_cnf"][type_index], sp["y_cnf"][type_index]]]),
                            ],
                            axis=0,
                        )
                        contour_with_z = np.concatenate(
                            [
                                contour_plus_xy,
                                np.repeat(event["z_dv_corr"], self.n_poly + 2)[:, np.newaxis],
                            ],
                            axis=1,
                        )
                        delta_r = self.map(contour_with_z)
                        scale = delta_r / np.linalg.norm(contour_with_z[:, :2], axis=1) + 1
                        scaled_2d_contour = contour_with_z[:, :2] * scale[:, np.newaxis]

                        # Store corrected contour and positions
                        result["position_contour_cnf"][event_i] = scaled_2d_contour[:-1]
                        result["x_cnf_fdc"][event_i] = scaled_2d_contour[-1, 0]
                        result["y_cnf_fdc"][event_i] = scaled_2d_contour[-1, 1]
                    else:
                        # Apply simple scaling based on field distortion correction
                        scale = event["r_field_distortion_correction"] / event["r_naive"] + 1
                        result["position_contour_cnf"][event_i] = (
                            sp["position_contours_cnf"][type_index] * scale
                        )
                        result["x_cnf_fdc"][event_i] = sp["x_cnf"][type_index] * scale
                        result["y_cnf_fdc"][event_i] = sp["y_cnf"][type_index] * scale

        return result


@export
class EventPositionUncertainty(strax.Plugin):
    """Plugin to calculate position uncertainties for events using the conditional normalising flow
    model. For information on the model, see note_.

    .. _note: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:juehang:flow_posrec_proposal_sr2 # noqa: E501

    This plugin computes uncertainties in radial (r) and angular (theta) positions
    for both main and alternative S2 signals. It uses the position contours
    calculated by a flow-based algorithm to estimate these uncertainties.

    The plugin provides:
    - Naive flow position uncertainties for main and alternative S2 signals
    - Corrected flow position uncertainties accounting for field distortions

    Depends on:
    - 'events': Basic event information
    - 'event_info': Additional event information
    - 'event_position_contour': Position contours for events

    Provides:
    - 'event_position_uncertainty': Calculated position uncertainties

    """

    __version__ = "0.0.2"

    depends_on = ("event_info", "event_position_contour")
    provides = "event_position_uncertainty"

    def infer_dtype(self):
        # Define the data types for the output
        infoline = {
            "s1": "main S1",
            "s2": "main S2",
            "alt_s1": "alternative S1",
            "alt_s2": "alternative S2",
        }
        dtype = []
        # Add fields for position uncertainties
        ptypes = ["s2", "alt_s2"]
        for type_ in ptypes:
            dtype += [
                (
                    (
                        f"Naive flow position uncertainty in r for {infoline[type_]}",
                        f"{type_}_r_position_uncertainty",
                    ),
                    np.float32,
                ),
                (
                    (
                        f"Naive flow position uncertainty in theta for {infoline[type_]}",
                        f"{type_}_theta_position_uncertainty",
                    ),
                    np.float32,
                ),
            ]

        # Add fields for corrected position uncertainties
        dtype += [
            (
                ("Flow position uncertainty in r (cm)", "r_position_uncertainty"),
                np.float32,
            ),
            (
                ("Flow position uncertainty in theta (rad)", "theta_position_uncertainty"),
                np.float32,
            ),
        ]
        # Add time fields
        dtype += strax.time_fields
        return dtype

    def compute(self, events):
        # Initialize the result array
        result = np.zeros(len(events), dtype=self.dtype)
        result["time"] = events["time"]
        result["endtime"] = strax.endtime(events)

        # Initialize uncertainty fields with NaN
        for type_ in ["s2", "alt_s2"]:
            result[f"{type_}_r_position_uncertainty"] *= np.nan
            result[f"{type_}_theta_position_uncertainty"] *= np.nan
        result["r_position_uncertainty"] *= np.nan
        result["theta_position_uncertainty"] *= np.nan

        # Calculate uncertainties for main and alternative S2 signals
        for type_ in ["s2", "alt_s2"]:
            # Calculate radial uncertainties
            r_array = np.linalg.norm(events[f"{type_}_position_contour_cnf_naive"], axis=2)
            r_min = np.min(r_array, axis=1)
            r_max = np.max(r_array, axis=1)

            # Calculate angular uncertainties
            theta_array = np.arctan2(
                events[f"{type_}_position_contour_cnf_naive"][..., 1],
                events[f"{type_}_position_contour_cnf_naive"][..., 0],
            )

            # Calculate average theta in contour to rotate contour by for correction
            avg_theta = np.arctan2(
                np.mean(events[f"{type_}_position_contour_cnf_naive"][..., 1], axis=1),
                np.mean(events[f"{type_}_position_contour_cnf_naive"][..., 0], axis=1),
            )

            # Correction for circular nature of angle (going over pi and -pi boundary)
            avg_theta = np.reshape(avg_theta, (avg_theta.shape[0], 1))
            theta_array_shift = (np.subtract(theta_array, avg_theta) + np.pi) % (2 * np.pi)
            theta_min = np.min(theta_array_shift, axis=1)
            theta_max = np.max(theta_array_shift, axis=1)

            theta_diff = theta_max - theta_min

            # Store uncertainties
            result[f"{type_}_r_position_uncertainty"] = (r_max - r_min) / 2
            result[f"{type_}_theta_position_uncertainty"] = np.abs(theta_diff) / 2

        # Apply field distortion correction to uncertainties
        scale = events["r_field_distortion_correction"] / events["r_naive"] + 1
        result["r_position_uncertainty"] = result["s2_r_position_uncertainty"] * scale
        result["theta_position_uncertainty"] = result["s2_theta_position_uncertainty"] * scale

        return result
