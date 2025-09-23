
import numpy as np
import straxen
import strax
from straxen.plugins.defaults import DEFAULT_POSREC_ALGO

export, __all__ = strax.exporter()

@export
class EventsPrimaryInfo(strax.Plugin):
    """
    Match primary MC interaction info to events by timing.

    For each event:
      * take clusters from microphysics_summary contained in the event
      * sum 'ed' of each cluster to get ed_tot
    """

    __version__ = "0.1.0"

    depends_on = ("microphysics_summary", "event_basics")
    provides = "events_primary_info"
    data_kind = "events"
    save_when = strax.SaveWhen.TARGET

    dtype = [
        (("Primary interaction x [cm]", "x_pri"), np.float32),
        (("Primary interaction y [cm]", "y_pri"), np.float32),
        (("Primary interaction z [cm]", "z_pri"), np.float32),
        (("Total deposited energy at primary [keV]", "ed_total"), np.float32),
        *strax.time_fields,
    ]

    def compute(self, interactions_in_roi, events, **kwargs):

        clusters_per_event = strax.split_by_containment(interactions_in_roi, events)

        result = np.zeros(len(events), dtype=self.dtype)
        result["x_pri"][:] = np.nan
        result["y_pri"][:] = np.nan
        result["z_pri"][:] = np.nan
        result["ed_total"][:] = np.nan

        result["time"] = events["time"]
        result["endtime"] = events["endtime"]

        for f in ("x_pri", "y_pri", "z_pri", "ed"):
            if f not in interactions_in_roi.dtype.names:
                raise ValueError(
                    f"Field '{f}' missing from interactions_in_roi; got "
                    f"{interactions_in_roi.dtype.names}"
                )

        for i, cl in enumerate(clusters_per_event):
            if len(cl) == 0:
                continue
            
            n_cl = len(cl)
            for n in range(n_cl):
                if (cl["x_pri"][n] != cl["x_pri"][0]) or (cl["y_pri"][n] != cl["y_pri"][0]) or (cl["z_pri"][n] != cl["z_pri"][0]):
                    print(f"Warning: Event {i} has clusters with different primary coordinates:")

            result["x_pri"][i] = cl["x_pri"][0]
            result["y_pri"][i] = cl["y_pri"][0]
            result["z_pri"][i] = cl["z_pri"][0]

            result["ed_total"][i] = np.nansum(cl["ed"]).astype(np.float32)

        return result