import numpy as np
import straxen
import numba
import typing as ty

import strax

export, __all__ = strax.exporter()


@export
class nVETOEventPositions(strax.Plugin):
    """Plugin which computes the interaction position in the nveto as an
    azimuthal angle."""

    __version__ = "0.1.1"

    depends_on = ("events_nv", "hitlets_nv")
    data_kind = "events_nv"
    provides = "event_positions_nv"
    compressor = "zstd"

    position_max_time_nv = straxen.URLConfig(
        default=20,
        infer_type=False,
        help="Time [ns] within an event use to compute the azimuthal angle of the " "event.",
    )

    nveto_pmt_position_map = straxen.URLConfig(
        help="nVeto PMT position mapfile",
        default="resource://nveto_pmt_position.csv?fmt=csv",
        infer_type=False,
    )

    def infer_dtype(self):
        return veto_event_positions_dtype()

    def setup(self):
        npmt_pos = self.nveto_pmt_position_map
        # Use records instead of a dataframe.
        self.pmt_properties = npmt_pos.to_records(index=False)

    def compute(self, events_nv, hitlets_nv):
        event_angles = np.zeros(len(events_nv), dtype=self.dtype)

        # Split hitlets by containment, works since we updated event start/end in
        # compute_event_properties.
        hits_in_events = strax.split_by_containment(hitlets_nv, events_nv)

        # Compute hitlets within the first x ns of event:
        hits_in_events, n_prompt = first_hitlets(hits_in_events, self.position_max_time_nv)
        event_angles["n_prompt_hitlets"] = n_prompt

        # Compute azimuthal angle and xyz positions:
        angle = get_average_angle(hits_in_events, self.pmt_properties)
        event_angles["angle"] = angle
        compute_positions(event_angles, hits_in_events, self.pmt_properties)
        strax.copy_to_buffer(events_nv, event_angles, f"_copy_events_nv")

        return event_angles


def veto_event_positions_dtype() -> list:
    dtype = []
    dtype += strax.time_fields
    dtype += [
        (
            (
                'Number of prompt hitlets within the first "position_max_time_nv" ns of the event.',
                "n_prompt_hitlets",
            ),
            np.int16,
        ),
        (
            ("Azimuthal angle, where the neutron capture was detected in [0, 2 pi).", "angle"),
            np.float32,
        ),
        (("Area weighted mean of position in x [mm]", "pos_x"), np.float32),
        (("Area weighted mean of position in y [mm]", "pos_y"), np.float32),
        (("Area weighted mean of position in z [mm]", "pos_z"), np.float32),
        (("Weighted variance of position in x [mm]", "pos_x_spread"), np.float32),
        (("Weighted variance of position in y [mm]", "pos_y_spread"), np.float32),
        (("Weighted variance of position in z [mm]", "pos_z_spread"), np.float32),
    ]
    return dtype


@numba.njit(cache=True, nogil=True)
def compute_positions(
    event_angles: np.ndarray,
    contained_hitlets: numba.typed.typedlist.List,
    pmt_pos: np.ndarray,
    start_channel: int = 2000,
):
    """Function which computes some artificial event position for a given
    neutron/muon-veto event. The position is computed based on a simple area
    weighted mean. Please note that the event position can be reconstructed in
    unphysical regions like being within the TPC.

    :param event_angles: Result array of the veto_event_position dtype.
        The result is updated inplace.
    :param contained_hitlets: Hitlets contained in each event.
    :param pmt_pos: Position of the veto PMTs
    :param start_channel: Starting channel of the detector.
    """
    for e_angles, hitlets in zip(event_angles, contained_hitlets):
        prompt_event_area = np.sum(hitlets["area"])
        if prompt_event_area:
            ch = hitlets["channel"] - start_channel
            pos_x = pmt_pos["x"][ch]
            pos_y = pmt_pos["y"][ch]
            pos_z = pmt_pos["z"][ch]

            e_angles["pos_x"] = np.sum(pos_x * hitlets["area"]) / prompt_event_area
            e_angles["pos_y"] = np.sum(pos_y * hitlets["area"]) / prompt_event_area
            e_angles["pos_z"] = np.sum(pos_z * hitlets["area"]) / prompt_event_area
            w = hitlets["area"] / prompt_event_area  # normalized weights
            if len(hitlets) and np.sum(w) > 0:
                e_angles["pos_x_spread"] = np.sqrt(
                    np.sum(w * (pos_x - e_angles["pos_x"]) ** 2) / np.sum(w)
                )
                e_angles["pos_y_spread"] = np.sqrt(
                    np.sum(w * (pos_y - e_angles["pos_y"]) ** 2) / np.sum(w)
                )
                e_angles["pos_z_spread"] = np.sqrt(
                    np.sum(w * (pos_z - e_angles["pos_z"]) ** 2) / np.sum(w)
                )


@numba.njit(cache=True, nogil=True)
def get_average_angle(
    hitlets_in_event: numba.typed.typedlist.List,
    pmt_properties: np.ndarray,
    start_channel: int = 2000,
) -> np.ndarray:
    """Computes azimuthal angle as an area weighted mean over all hitlets.

    :param hitlets_in_event: numba.typed.List containing the hitlets per
        event.
    :param pmt_properties: numpy.sturctured.array containing the PMT
        positions in the fields "x" and "y".
    :param start_channel: First channel e.g. 2000 for nevto.
    :return: np.array holding the azimuthal angles.
    """
    res = np.zeros(len(hitlets_in_event), np.float32)
    for ind, hitlets in enumerate(hitlets_in_event):
        if np.sum(hitlets["area"]):
            x = pmt_properties["x"][hitlets["channel"] - start_channel]
            y = pmt_properties["y"][hitlets["channel"] - start_channel]

            weighted_mean_x = np.sum(x * hitlets["area"]) / np.sum(hitlets["area"])
            weighted_mean_y = np.sum(y * hitlets["area"]) / np.sum(hitlets["area"])
            res[ind] = _circ_angle(weighted_mean_x, weighted_mean_y)
        else:
            res[ind] = np.nan
    return res


@numba.njit(cache=True, nogil=True)
def circ_angle(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    """Loops over a set of x and y values and computes azimuthal angle.

    :param x_values: x-coordinates
    :param y_values: y-coordinates
    :return: angles
    """
    res = np.zeros(len(x_values), dtype=np.float32)
    for ind, (x, y) in enumerate(zip(x_values, y_values)):
        res[ind] = _circ_angle(x, y)
    return res


@numba.njit(cache=True, nogil=True)
def _circ_angle(x: float, y: float) -> float:
    if x > 0 and y >= 0:
        # 1st quadrant
        angle = np.abs(np.arctan(y / x))
        return angle
    elif x <= 0 and y > 0:
        # 2nd quadrant
        angle = np.abs(np.arctan(x / y))
        return np.pi / 2 + angle
    elif x < 0 and y <= 0:
        # 3rd quadrant
        angle = np.abs(np.arctan(y / x))
        return np.pi + angle
    elif y < 0 and x >= 0:
        # 4th quadrant
        angle = np.abs(np.arctan(x / y))
        return 3 / 2 * np.pi + angle
    elif x == 0 and y == 0:
        return np.NaN
    else:
        print(x, y)
        raise ValueError("It should be impossible to arrive here, " "but somehow we managed.")


@numba.njit(cache=True, nogil=True)
def first_hitlets(
    hitlets_per_event: np.ndarray, max_time: int
) -> ty.Tuple[numba.typed.List, np.ndarray]:
    """Returns hitlets within the first "max_time" ns of an event.

    :param hitlets_per_event: numba.typed.List of hitlets per event.
    :param max_time: int max allowed time difference to leading hitlet
        in ns.
    """
    res_hitlets_in_event = numba.typed.List()
    res_n_prompt = np.zeros(len(hitlets_per_event), np.int16)
    for ind, hitlets in enumerate(hitlets_per_event):
        m = (hitlets["time"] - hitlets[0]["time"]) < max_time
        h = hitlets[m]
        res_hitlets_in_event.append(h)
        res_n_prompt[ind] = len(h)
    return res_hitlets_in_event, res_n_prompt
