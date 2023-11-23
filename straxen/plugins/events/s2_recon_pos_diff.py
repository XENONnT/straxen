import numpy as np
import straxen
import strax

export, __all__ = strax.exporter()


@export
class S2ReconPosDiff(strax.Plugin):
    """Plugin that provides position reconstruction difference for S2s in events, see note:

    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:shengchao:sr0:reconstruction_quality

    """

    __version__ = "0.0.3"

    parallel = True
    depends_on = "event_basics"
    provides = "s2_recon_pos_diff"
    save_when = strax.SaveWhen.EXPLICIT

    recon_alg_included = straxen.URLConfig(
        help="The list of all reconstruction algorithm considered.",
        default=("_mlp", "_gcn", "_cnn"),
        infer_type=False,
    )

    def infer_dtype(self):
        dtype = [
            ("s2_recon_avg_x", np.float32, "Mean value of x for main S2"),
            ("alt_s2_recon_avg_x", np.float32, "Mean value of x for alternatice S2"),
            ("s2_recon_avg_y", np.float32, "Mean value of y for main S2"),
            ("alt_s2_recon_avg_y", np.float32, "Mean value of y for alternatice S2"),
            ("s2_recon_pos_diff", np.float32, "Reconstructed position difference for main S2"),
            (
                "alt_s2_recon_pos_diff",
                np.float32,
                "Reconstructed position difference for alternative S2",
            ),
        ]
        dtype += strax.time_fields
        return dtype

    def compute(self, events):
        result = np.zeros(len(events), dtype=self.dtype)
        result["time"] = events["time"]
        result["endtime"] = strax.endtime(events)
        # Computing position difference
        self.compute_pos_diff(events, result)
        return result

    def cal_avg_and_std(self, values, axis=1):
        average = np.mean(values, axis=axis)
        std = np.std(values, axis=axis)
        return average, std

    def eval_recon(self, data, name_x_list, name_y_list):
        """This function reads the name list based on s2/alt_s2 and all recon algorithm registered
        Each row consists the reconstructed x/y and their average and standard deviation is
        calculated."""
        # lazy fix to delete field name in array, otherwise np.mean will complain
        x_avg, x_std = self.cal_avg_and_std(np.array(data[name_x_list].tolist()))
        y_avg, y_std = self.cal_avg_and_std(np.array(data[name_y_list].tolist()))
        r_std = np.sqrt(x_std**2 + y_std**2)
        res = x_avg, y_avg, r_std
        return res

    def compute_pos_diff(self, events, result):
        alg_list = self.recon_alg_included
        for peak_type in ["s2", "alt_s2"]:
            # Selecting S2s for pos diff
            # - must exist (index != -1)
            # - must have positive AFT
            # - must contain all alg info
            cur_s2_bool = events[peak_type + "_index"] != -1
            cur_s2_bool &= events[peak_type + "_area_fraction_top"] > 0
            for name in self.recon_alg_included:
                cur_s2_bool &= ~np.isnan(events[peak_type + "_x" + name])
                cur_s2_bool &= ~np.isnan(events[peak_type + "_y" + name])

            # default value is nan, it will be overwrite if the event satisfy the requirements
            result[peak_type + "_recon_pos_diff"][:] = np.nan
            result[peak_type + "_recon_avg_x"][:] = np.nan
            result[peak_type + "_recon_avg_y"][:] = np.nan

            if np.any(cur_s2_bool):
                name_x_list = []
                name_y_list = []
                for alg in alg_list:
                    name_x_list.append(peak_type + "_x" + alg)
                    name_y_list.append(peak_type + "_y" + alg)

                # Calculating average x,y, and position difference
                x_avg, y_avg, r_std = self.eval_recon(events[cur_s2_bool], name_x_list, name_y_list)
                result[peak_type + "_recon_pos_diff"][cur_s2_bool] = r_std
                result[peak_type + "_recon_avg_x"][cur_s2_bool] = x_avg
                result[peak_type + "_recon_avg_y"][cur_s2_bool] = y_avg
