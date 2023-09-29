from typing import Union, Tuple
from immutabledict import immutabledict

import numpy as np
import holoviews as hv


# noinspection PyArgumentList
class PlotPeakLikeData:
    """Base class to plot peak like data for any type of detector."""

    # Properties to be overwritten by sub-clases if needed.
    _xlabel = "x-label"  # Label for x axis defined in _set_labels
    _ylabel = "y-label"  # Label for y axis defined in _set_labels
    _label = "label"  # Label for plot defined in _set_labels
    _scaler = 1
    keep_amplitude_per_sample = False
    _vdim_labels = immutabledict()
    _never_include_fields: Union[Tuple[str, ...], Tuple[()]] = ()

    def __init__(
        self,
        opts=immutabledict(),
    ):
        """
        :param opts_dict: Dictionary of options which will be applied to
            hv.Area and hv.Curve plots. If you want to change a setting
            which is not common by both plot types please specify them
            in "plot_peak" or "plot_peaks".
        """
        if not isinstance(opts, (dict, immutabledict)):
            raise ValueError('"opts_dict" must be a dictionary!')
        self.opts_dict = opts

    def plot_peak(self, peak):
        """User defined function for plotting a single peak with correct labels etc."""
        raise NotImplementedError()

    def plot_peaks(self, peaks):
        """User defined function for plotting multiple peaks into the same figure."""
        raise NotImplementedError()

    def _set_labels(self, peaks, peak_type, time_prefix):
        """User defined function which should set the xlabel, ylabel and plot label for our peak
        plots.

        Also should define the scaler which scales the time axis in desired unit e.g. µs or ns.

        """
        raise NotImplementedError()

    def _plot_peak(
        self,
        peak,
        label,
        group_label="Peak",
        opts_curve=immutabledict(),
        opts_area=immutabledict(),
        _relative_start_time=0,
    ):
        """Plots area and curve to display a peak. :param peak: Peak to be
        displayed :param label: Label to be used in legend. :param opts_curve:
        Additional options to be applied to hv.Curve :param opts_area:
        Additional options to be applied to hv.Area :param
        _relative_start_time: Time to which the peak should be plotted
        relatively. E.g. event start time.

        :returns: hv.Area and hv.Curve
        """
        data = self.get_peak_data(
            peak,
            relative_start=_relative_start_time,
        )

        area = hv.Area(
            data,
            kdims=hv.Dimension("xs", label=self._xlabel),
            vdims=[
                hv.Dimension("ys", label=self._ylabel),
            ],
            label=label,
            group=group_label,
        ).opts(alpha=0.3, **opts_area, framewise=True)

        curve = hv.Curve(
            data,
            kdims=hv.Dimension("xs", label=self._xlabel),
            vdims=[
                hv.Dimension("ys", label=self._ylabel),
            ]
            + self._vdims,
            label=label,
            group=group_label,
        ).opts(**opts_curve, framewise=True)

        return area, curve

    def _init_peak_plot(self, peak, peak_type, time_prefix):
        """Initalizes empty plots for plotting single or muliple-peaks into the same figure.

        Intializes plot legend labels as well as other dimensions.

        """
        self._set_single_valued_dimensions(peak)
        self._set_labels(peak, peak_type, time_prefix)

        area = hv.Area(
            None,
            kdims=hv.Dimension("xs", label=self._xlabel),
            vdims=[
                hv.Dimension("ys", label=self._ylabel),
            ],
        ).opts(framewise=True)

        curve = hv.Curve(
            None,
            kdims=hv.Dimension("xs", label=self._xlabel),
            vdims=[
                hv.Dimension("ys", label=self._ylabel),
            ]
            + self._vdims,
        ).opts(framewise=True)
        return area, curve

    def _set_single_valued_dimensions(self, peak):
        """Extracts information from peaks. Defines a list of single valued dimensions which will be
        extracted for the plot in _get_peak_data.

        Defines holovies vdim Dimensions for a proper hover-tool display. Looks up unit describition
        in _vdim_labels which have to be specified by the user.

        """
        self.single_dimension_fields = self._get_single_dimension_fields(peak)
        vdims = []

        for field_name in self.single_dimension_fields:
            if field_name in self._never_include_fields:
                continue
            vdims.append(
                hv.Dimension(field_name, label=self._vdim_labels.get(field_name, field_name))
            )
        self._vdims = vdims

    @staticmethod
    def _get_single_dimension_fields(peak):
        """Function which returns all single valued qunatity names for a given peak."""
        field_names = []
        for field_name in peak.dtype.names:
            value = peak[field_name]
            _is_single_value_quantity = value.ndim == 1
            if _is_single_value_quantity:
                field_names.append(field_name)

        return field_names

    def _is_time_based_quantity(self, field_name):
        """Function which checks if specified parameter is a quantiy which is based on time, like
        range, width or time properties."""
        raise NotImplementedError()

    def get_peak_data(self, peak, relative_start=0):
        """Function which extracts base peaks information as a dictionary. Can be further customized
        in sub-classes if needed.

        :param peak: Peaks to be plotted. :param relative_start: t0 from which on the peaks should
        be     plotted. :return: dictionary

        """
        # Wrapper in case one hast to customized things further in one
        # of the sub-clases.
        return self._get_peak_data(peak, relative_start=relative_start)

    def _get_peak_data(self, peak, relative_start=0):
        """Function which extracts base peaks information as a dictionary. Extracts all information
        which are stored as single valued parameters.

        :param peaks: Peaks to be plotted. :param relative_start: t0 from which on the peaks should
        be     plotted. :return: dictionary

        """
        x, y = self._patches_x_y(peak)
        x -= relative_start  # relative to first peak
        x = x / self._scaler  # Scaler to convert ns to µs or other units

        # Coordinates for peak patches
        data = {
            "xs": x,
            "ys": y,
        }

        # Add other single valued fields which we can find to our
        # dictionary. Adjust time parameters according to scaling.
        for field_name in self.single_dimension_fields:
            value = peak[field_name]

            if self._is_time_based_quantity(field_name):
                data[field_name] = value / self._scaler
            else:
                data[field_name] = value

        return data

    def _patches_x_y(self, peak):
        """Creates x,y coordinates needed to make a stepwise function with hv.Areas.

        :param peak: Peak for which we need the x/y samples :returns: Tuple of x, y

        """
        if self.keep_amplitude_per_sample:
            dt_a = 1
        else:
            dt_a = peak["dt"]

        x = np.arange(peak["length"])
        xx = np.zeros(2 * len(x), dtype=x.dtype)
        mx = 0.5 * (x[1::] + x[:-1])
        xx[1:-1:2] = mx
        xx[2::2] = mx
        xx[0] = 1.5 * x[0] - 0.5 * x[1]
        xx[-1] = 1.5 * x[-1] - 0.5 * x[-2]
        xx = np.array(xx) * peak["dt"] + peak["time"]
        y = peak["data"][: peak["length"]]
        yy = np.zeros(2 * len(x))
        yy[0::2] = y
        yy[1::2] = y
        yy = np.array(yy) / dt_a

        # baseline since we'll fill underneath
        xx = np.concatenate([[xx[0]], xx, [xx[-1]]])
        yy = np.concatenate([[0], yy, [0]])
        return xx, yy


# noinspection PyArgumentList
class PlotPeaksTPC(PlotPeakLikeData):
    """Class which plots single or multiple peaks for interactive holoviews display."""

    time_in_us = False
    _never_include_fields = (
        "time",
        "endtime",
        "center_time",
        "channel",
        "type",
        "max_goodness_of_split",
    )
    # Use round brakets for units which should not be affected by
    # "time_in_µs".
    _vdim_labels = {
        "time": "time (ns)",
        "endtime": "endtime (ns)",
        "center_time": "center_time (ns)",
        "rise_time": "rise_time [ns]",
        "range_50p_area": "range_50p_area [ns]",
        "range_90p_area": "range_90p_area [ns]",
        "area": "area [PE]",
        "max_pmt_area": "max_pmt_area [PE]",
    }

    def __init__(
        self,
        opts=immutabledict(),
        keep_amplitude_per_sample=False,
    ):
        super().__init__(opts)
        self.keep_amplitude_per_sample = keep_amplitude_per_sample

    def _set_labels(self, peak, peak_type, time_prefix):
        if self.time_in_us:
            scaler = 1000
            xlabel = f"{time_prefix} Time [µs]"
            self._change_units()
        else:
            xlabel = f"{time_prefix} Time [ns]"
            scaler = 1

        if self.keep_amplitude_per_sample:
            ylabel = f"{peak_type} Amplitude [pe/sample]"
        else:
            ylabel = f"{peak_type} Amplitude [pe/ns]"

        self._scaler = scaler
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._label = f'S{peak["type"]}'

    def _change_units(self):
        for dim in self._vdims:
            dim.label = dim.label.replace("[ns]", "[µs]")

    def plot_peak(
        self,
        peak,
        opts_area=immutabledict(),
        opts_curve=immutabledict(),
        label="Peak",
        time_in_us=False,
        amplitude_prefix="",
    ):
        """Function which plots a single peak."""
        plot = self.plot_peaks(
            peak,
            opts_curve=opts_curve,
            opts_area=opts_area,
            time_in_us=time_in_us,
            label=label,
            group_label="Peak",
            amplitude_prefix=amplitude_prefix,
            _relative_start_time=peak["time"],
        )
        # FIY: If opts is applied only with an empty dict all previous
        # options seem to be reseted to default.
        return plot.opts(**self.opts_dict, legend_limit=10)

    def plot_peaks(
        self,
        peaks,
        label="Peaks",
        group_label="Event",
        opts_area=immutabledict(),
        opts_curve=immutabledict(),
        time_in_us=False,
        time_prefix="",
        amplitude_prefix="",
        _relative_start_time=0,
    ):
        if type(peaks) is np.void:
            # In case of supplying only a single peak we have to make
            # an array again:
            peaks = np.array([peaks])

        if not (np.all(peaks["type"] == peaks[0]["type"])):
            raise ValueError("All peaks must be of the same type (S1, S2 or Unknown)!")

        self.time_in_us = time_in_us
        area, curve = self._init_peak_plot(peaks, amplitude_prefix, time_prefix)
        for ind, peak in enumerate(peaks):
            _area, _curve = self._plot_peak(
                peak,
                label=label + f"_{ind}",
                group_label=group_label,
                opts_curve=opts_curve,
                opts_area=opts_area,
                _relative_start_time=_relative_start_time,
            )
            area *= _area
            curve *= _curve.opts(tools=["vline"])
        return (area * curve).opts(**self.opts_dict, legend_limit=100)

    def _is_time_based_quantity(self, field_name):
        """Checks if fiel_name belongs to a time based peak property to rescale ns value to µs for
        S2s."""
        _is_true = "time" in field_name and (
            (field_name != "time") and (field_name != "endtime") and (field_name != "center_time")
        )
        _is_true |= "width" in field_name
        _is_true |= "range" in field_name
        return _is_true
