import strax
import straxen
from straxen.analyses.holoviews_waveform_display import seconds_from

import panel as pn
from bokeh.models import HoverTool

import numpy as np
import pandas as pd

export, __all__ = strax.exporter()


@export
class nVETOEventDisplay:
    def __init__(
        self,
        events=None,
        hitlets=None,
        run_id=0,
        channel_range=(2000, 2119),
        pmt_map="nveto_pmt_position.csv",
        plot_extension="bokeh",
    ):
        """Class to plot an interactive nveto display.

        :param events: Events which should be plot. Can also be none in case the hitlet matrix
            and/or pattern map should be plotted separately.
        :param hitlets: Same as events, but hitlets_nv.
        :param run_id: Run_id which should be displayed in the title.
        :param channel_range: Channel range of the detector.
        :param pmt_map: PMT map which is loaded via straxen.get_resource. The map has to contain the
            channel number, and xyz coordinates.
        :param plot_extension: Extension which should be used for rendering can be either bokeh or
            matpltolib. Default is bokeh to support dynamic plots.

        """
        self.import_holoviews()
        self.hv.extension(plot_extension)
        self.df_event_time = None
        self.df_event_properties = None
        self.hitlets = hitlets
        self.channel_range = channel_range
        self.run_id = run_id

        # Load PMT data:
        if isinstance(pmt_map, str):
            self.pmt_positions = straxen.get_resource(pmt_map, fmt="csv")
        elif isinstance(pmt_map, np.ndarray):
            self.pmt_positions = pmt_map
        else:
            raise ValueError(
                "pmt_map not understood, has either to be "
                f'a string or a numpy array, got "{pmt_map}".'
            )

        if events is not None:
            self.event_df = straxen.convert_array_to_df(events)
        else:
            self.event_df = None

        if events is not None and hitlets is not None:
            self.hitlets_per_event = strax.split_by_containment(hitlets, events)

    def import_holoviews(self):
        import holoviews as hv

        self.hv = hv

    def plot_event_display(self):
        """Creates an interactive event display for the neutron veto.

        :return: panel.Column hosting the plots and panels.

        """

        # First we have to define the python callbacks:
        def matrix_callback(value):
            """Callback for the dynamic hitlet matrix.

            Changes polygons when a new event is selected.

            """
            self.hitlet_points = self.hitlets_to_hv_points(
                self.hitlets_per_event[value], t_ref=self.event_df.loc[value, "time"]
            )

            # Create the hitlet matrix and time stream:
            self.hitlet_matrix = self.plot_hitlet_matrix(
                hitlets=None, _hitlet_points=self.hitlet_points
            )
            return self.hitlet_matrix

        def pattern_callback(value, x_range):
            """Call back for the dynamic PMT pattern map.

            Depends on the selcted event as well as the selected x_range in the hitlet_matrix.

            """
            # Get hitlet points and select only points within x_range:
            hit = self.hitlet_points.data
            if not x_range:
                time = hit["time"].values
                length = hit["length"].values
                dt = hit["dt"].values
                x_range = [min(time), max(time + (length * dt))]

            m = (hit["time"] >= x_range[0]) & (hit["time"] < x_range[1])

            hitlets_in_time = hit[m]
            new_points = self.hv.Points(hitlets_in_time)

            # Plot pmt pattern:
            pmts = self.plot_nveto(
                hitlets=None, _hitlet_points=new_points, pmt_size=8, pmt_distance=0.5
            )
            angle = self._plot_reconstructed_position(value)
            return angle * pmts

        self._make_sliders_and_tables(self.event_df)
        index = self.evt_sel_slid.value
        self.hitlet_points = self.hitlets_to_hv_points(
            self.hitlets_per_event[index], t_ref=self.event_df.loc[index, "time"]
        )

        dmap_hitlet_matrix = self.hv.DynamicMap(
            matrix_callback, streams=[self.evt_sel_slid.param.value]
        ).opts(framewise=True)

        time_stream = self.hv.streams.RangeX(source=dmap_hitlet_matrix)

        dmap_pmts = self.hv.DynamicMap(
            pattern_callback, streams=[self.evt_sel_slid.param.value, time_stream]
        )

        slider_column = pn.Column(
            self.evt_sel_slid, self.evt_sel_slid.controls(["value"]), self.time_table
        )

        event_display = pn.Column(
            self.title_panel,
            pn.Row(slider_column, dmap_pmts, width_policy="max"),
            dmap_hitlet_matrix,
            self.prop_table,
            self.pos_table,
            width_policy="max",
        )
        return event_display

    def plot_hitlet_matrix(self, hitlets, _hitlet_points=None):
        """Function which plots the hitlet matrix for the specified hitlets. The hitlet matrix is
        something equivalent to the record matrix for the TPC.

        :param hitlets: Hitlets to be plotted if called directly.
        :param _hitlet_points: holoviews.Points created by the event display. Only internal use.
        :return: hv.Polygons plot.

        """
        if not _hitlet_points:
            _hitlet_points = self.hitlets_to_hv_points(
                hitlets,
            )

        hitlet_matrix = self._plot_base_matrix(_hitlet_points).opts(
            title="Hitlet Matrix",
            xlabel="Time [ns]",
            ylabel="PMT channel",
            ylim=(1995, 2125),
            color="area",
            clabel="Area [pe]",
            cmap="viridis",
            colorbar=True,
        )
        return hitlet_matrix

    def _plot_base_matrix(self, hv_point_data):
        """Base function to plot record or hitlet matrix."""
        matrix_plot = self.hv.Segments(
            hv_point_data.data, kdims=["time", "channel", "endtime", "channel"]
        ).opts(
            tools=["hover"],
            aspect=4,
            responsive="width",
        )
        return matrix_plot

    def plot_nveto(
        self,
        hitlets,
        pmt_size=8,
        pmt_distance=0.5,
        _hitlet_points=None,
    ):
        """Plots the nveto pmt pattern map for the specified hitlets. Expects hitlets to be sorted
        in time.

        :param hitlets: Hitlets to be plotted if called directly.
        :param pmt_size: Base size of a PMT for 1 pe.
        :param pmt_distance: Scaling parameter for the z -> xy projection.
        :param _hitlet_points: holoviews.Points created by the event display. Only internal use.
        :return: stacked hv.Points plot.

        """
        if not _hitlet_points:
            _hitlet_points = self.hitlets_to_hv_points(
                hitlets,
            )

        pmts = self._plot_nveto(_hitlet_points.data, pmt_size=pmt_size, pmt_distance=pmt_distance)

        # Plot channels which have not seen anything:
        pmts_data = pmts.data
        m = pmts_data.loc[:, "area"].values == 0
        blanko_pmts = np.zeros(np.sum(m), dtype=strax.hitlet_dtype())
        blanko_pmts["area"] = 1
        blanko_pmts["channel"] = pmts_data.loc[m, "channel"]
        blanko = self._plot_nveto(blanko_pmts, pmt_size=pmt_size, pmt_distance=pmt_distance)

        blanko = blanko.opts(
            fill_color="white", tools=[HoverTool(tooltips=[("PMT", "@channel")])], color=None
        )

        detector_layout = (
            plot_tpc_circle(straxen.cryostat_outer_radius)
            * plot_diffuser_balls_nv()
            * plot_nveto_reflector()
        )

        return detector_layout * blanko * pmts

    def _plot_nveto(self, pmts, pmt_size, pmt_distance):
        if isinstance(pmts, pd.DataFrame):
            pmts = pmts.to_records(index=False)
        hv = self.hv
        # Get hitlet_points data and extract PMT data
        pmt_data = self._create_nveto_pmts(pmts, pmt_size, pmt_distance)

        pmts = hv.Points(
            data=pd.DataFrame(pmt_data),
            kdims=[hv.Dimension("pmt_x", label="x [cm]"), hv.Dimension("pmt_y", label="y [cm]")],
            vdims=["time", "size", "area", "channel"],
        ).opts(
            size="size",
            tools=[self._create_pmt_pattern_hover()],
            color="time",
            colorbar=True,
            cmap="viridis",
            clabel="Time [ns]",
            line_color="black",
            alpha=0.8,
            width=400,
            title="nVETO Top View",
        )

        return pmts

    def _create_nveto_pmts(
        self,
        hitlets,
        pmt_size,
        pmt_distance,
        max_area_scale=10,
    ):
        """Function which creates data for the nVTEO PMT points of the pattern plot."""
        # Get PMT xy-position based on projection function:
        pmt_data = self._convert_channel_to_xy(pmt_distance=pmt_distance)

        for h in hitlets:
            ch = h["channel"] - self.channel_range[0]
            pmt_data[ch]["area"] += h["area"]

            if not pmt_data[ch]["time"]:
                # Get first arrival time for channel:
                pmt_data[ch]["time"] = h["time"]

        pmt_data["size"] = pmt_data["area"] * pmt_size
        pmt_data["size"] = np.clip(pmt_data["size"], 0, max_area_scale * pmt_size)

        return pmt_data

    def _convert_channel_to_xy(self, pmt_distance=0.5):
        """Projects PMT channel from x/y/z onto xy."""
        pmt_data = np.zeros(
            120,
            dtype=[
                ("pmt_x", np.float64),
                ("pmt_y", np.float64),
                ("channel", np.int32),
                ("column", np.int32),
                ("area", np.float64),
                ("size", np.float64),
                ("time", np.float64),
            ],
        )
        pmt_data["pmt_x"] = self.pmt_positions["x"]
        pmt_data["pmt_y"] = self.pmt_positions["y"]
        pmt_data["channel"] = self.pmt_positions["channel"]
        pmt_data["column"] = np.repeat(np.arange(20), 6)

        z = self.pmt_positions["z"]
        z_min = self.pmt_positions["z"].min()
        z_distance = self.pmt_positions["z"].max() - z_min
        scale = (z - z_min) / z_distance * pmt_distance

        # Only correct x or y depending on column position:
        m_x = np.isin(pmt_data["column"], (4, 5, 6, 7, 8, 14, 15, 16, 17, 18))
        m_y = np.isin(pmt_data["column"], (19, 0, 1, 2, 3, 9, 10, 11, 12, 13))

        pmt_data["pmt_x"][m_x] = pmt_data["pmt_x"][m_x] * (1 + scale[m_x])
        pmt_data["pmt_y"][m_y] = pmt_data["pmt_y"][m_y] * (1 + scale[m_y])

        return pmt_data

    @staticmethod
    def _create_pmt_pattern_hover():
        tooltips = [
            ("rel. time [ns]", "@time"),
            ("integrated area [pe]", "@area"),
            ("PMT", "@channel"),
        ]
        return HoverTool(tooltips=tooltips)

    @staticmethod
    def hitlets_to_hv_points(
        hitlets,
        t_ref=None,
    ):
        """Function which converts hitlets into hv.Points used in the different plots.

        Computes hitlet times as relative times with respect to the first hitlet if t_ref is not
        set.

        """
        import holoviews as hv

        if not len(hitlets):
            raise ValueError("Expected at least a single hitlet.")

        if isinstance(hitlets, np.ndarray):
            hitlets = pd.DataFrame(hitlets)

        # Set relative times:
        if t_ref is None:
            t_ref = min(hitlets["time"])

        time = seconds_from(hitlets["time"], t_ref, unit_conversion=1)
        hitlets["time"] = time
        hitlets["endtime"] = strax.endtime(hitlets.to_records())

        hitlet_points = hv.Points(hitlets)

        return hitlet_points

    def _plot_reconstructed_position(self, index):
        """Function which plots the nVETO event position according to its azimuthal angle.

        :param index: Which event to plot.

        """

        x = (0, np.real(np.exp(self.event_df.loc[index, "angle"] * 1j)) * 400)
        y = (0, np.imag(np.exp(self.event_df.loc[index, "angle"] * 1j)) * 400)
        angle = self.hv.Curve((x, y)).opts(
            color="orange", line_dash="dashed", xlim=(-350, 350), ylim=(-350, 350)
        )
        return angle

    def _make_sliders_and_tables(self, df):
        """Function which creates interactive sliders and tables for the neutron-veto event
        display."""
        if not len(df):
            raise ValueError("DataFrame must be at least one entry long.")

        self.evt_sel_slid = pn.widgets.IntSlider(value=0, start=0, end=len(df))
        self._make_tables(df)

        # Define callbacks for tables:
        self.evt_sel_slid.param.watch(
            lambda event: table_callback(self.time_table, self.df_event_time, event, True), "value"
        )
        self.evt_sel_slid.param.watch(
            lambda event: table_callback(self.prop_table, self.df_event_properties, event), "value"
        )
        self.evt_sel_slid.param.watch(
            lambda event: table_callback(self.pos_table, self.df_event_position, event), "value"
        )

        # Now make title and also define callback:
        title = self._make_title(self.evt_sel_slid.value)
        self.title_panel = pn.panel(title, sizing_mode="scale_width")

        def title_callback(event):
            self.title_panel.object = self._make_title(event.new)

        self.evt_sel_slid.param.watch(title_callback, "value")

    def _make_tables(self, df):
        """Divides event data into a time, properties and position table."""
        # Time table:
        time_keys = ["time", "endtime", "event_number_nv"]
        self.df_event_time = df.loc[:, time_keys]

        # Properties tables:
        pos_keys = [
            "angle",
            "pos_x",
            "pos_x_spread",
            "pos_y",
            "pos_y_spread",
            "pos_z",
            "pos_z_spread",
        ]
        self.df_event_position = df.loc[:, pos_keys]

        keys = df.columns.values
        keys = [k for k in keys if k not in time_keys + pos_keys]
        self.df_event_properties = df.loc[:, keys]

        # Table panels:
        index = self.evt_sel_slid.value
        self.time_table = pn.panel(
            self.df_event_time.loc[index],
        )
        self.pos_table = pn.panel(
            self.df_event_position.loc[index:index, :], sizing_mode="scale_width"
        )

        self.prop_table = pn.panel(
            self.df_event_properties.loc[index:index, :], sizing_mode="scale_width"
        )

    def _make_title(self, ind):
        """Function which creates title test in markdown format."""
        start = self.df_event_time.loc[ind, "time"]
        date = np.datetime_as_string(start.astype("<M8[ns]"), unit="s")
        start_ns = start - (start // straxen.units.s) * straxen.units.s
        end = self.df_event_time.loc[ind, "endtime"]
        end_ns = end - start + start_ns
        return "".join(
            (
                f"##Event {ind} from run {self.run_id}\n",
                f"##Recorded at ({date[:10]} {date[10:]}) UTC ",
                f"{start_ns} ns - {end_ns} ns",
            )
        )


def plot_tpc_circle(radius):
    """Plots TPC as a black circle.

    :param radius: Radius in cm.

    """
    import holoviews as hv

    x = radius * np.cos(np.arange(-np.pi, np.pi + 0.1, 0.01))
    y = radius * np.sin(np.arange(-np.pi, np.pi + 0.1, 0.01))
    return hv.Curve((x, y)).opts(color="k")


def plot_diffuser_balls_nv():
    """Computes position of nveto diffuser balls.

    :return: hv.Points with hover tool.

    """
    import holoviews as hv

    cryostat_r = straxen.cryostat_outer_radius
    depth_stiffening_ring = 8
    r = cryostat_r + depth_stiffening_ring
    angles = np.array([-8, 90, -90, 180])
    x = r * np.cos(angles / 180 * np.pi)
    y = r * np.sin(angles / 180 * np.pi)
    db_ids = (18, 17, 11, 15)
    data = pd.DataFrame(np.array([x, y, db_ids]).T, columns=("x", "y", "id"))
    return hv.Points(data=data).opts(size=8, tools=["hover"])


def plot_nveto_reflector():
    """Function which plots the lateral reflector panels of the neutron veto.

    Coordinates are based on the MC coordinates of the octagon
    model. The coordinates can be found in this note:
    id=xenon:xenonnt:mc:notes:nveto-geometry#lateral_reflectors

    """
    import holoviews as hv

    xy_center_angle = np.array(
        [
            (0, -1875.25, -90),
            (1442.28, -1442.28, -45),
            (1875.25, 0, 0),
            (1442.28, 1442.28, 45),
            (0, 1875.25, 90),
            (-1442.28, 1442.28, 135),
            (-1875.25, 0, 180),
            (-1441.75, -1442.28, -135),
            (0, -1875.25, -90),
        ],
        dtype=[
            ("x", np.float64),
            ("y", np.float64),
            ("phi", np.float64),
        ],
    )
    xy_pos = _compute_lateral_reflector_xy_edges(
        xy_center_angle, long_side_length=2018, short_side_length=1224
    )
    xy_pos = pd.DataFrame(xy_pos, columns=("x", "y"))

    return hv.Curve(data=xy_pos).opts(color="k")


def _compute_lateral_reflector_xy_edges(
    xy_center_angle, long_side_length=2018, short_side_length=1224
):
    """Function which computes the position of the lateral reflector panels. Input in mm return in
    cm.

    :param xy_center_angle: Center xy coordinate and angle of each panel.
    :param long_side_length: Full length of the long panels in mm.
    :param short_side_length: Same

    """
    res = np.zeros(len(xy_center_angle), dtype=[("x", np.float64), ("y", np.float64)])
    for ind, xy in enumerate(xy_center_angle):
        if ind % 2:
            length = short_side_length / 2
        else:
            length = long_side_length / 2
        xc, yc, phi = xy[["x", "y", "phi"]]
        x = xc - length * np.sin(phi * np.pi / 180)
        y = yc + length * np.cos(phi * np.pi / 180)
        res[ind]["x"] = x / 10
        res[ind]["y"] = y / 10

    return res


def table_callback(table, data, event, column=False):
    """Callback template for tables used together with a pn.widget.IntSlider.

    :param table: pn.Panel object of a pd.DataFrame
    :param data: pd.DataFrame storing the data.
    :param event: Slider event returned by param.watch
    :param column: Boolean if true uses horizontal columns to present data.

    """
    if column:
        table.object = data.loc[event.new]
    else:
        table.object = data.loc[event.new : event.new]
