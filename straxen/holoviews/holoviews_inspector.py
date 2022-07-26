import param
import panel as pn
import holoviews as hv
import numpy as np
from immutabledict import immutabledict

class EventStats(param.Parameterized):
    """Class which displays all events in a user defined data space. 
    Marks position of single selected event depending on the passed
    index or index widget.
    
    :param events: Events for which histogram should be plotted.
    :param default_config: Dictionary containing the default settings
        for the plot.
    :param exclude_fields: A list of fields to be excluded from the 
        x/y-selection widget. Can be uselful as event_info has a lot of
        fields. Can be either a list/tuple or a function. If passed as
        function must take as an argument a list with field names and 
        return a list with field names.
    :param _index_slider: Parameter to pass an external integer slider 
        to class. E.g. required in event display.
    
    Warning:
        In case the initally selected value is outside of the specified
        plotting range the plot will break.
    """
    _x_edges = None
    _y_edges = None
    _entries = None
    _x_field_name = None
    _y_field_name = None
    _x_bins = None
    _y_bins = None
    _x_log = False
    _y_log = False
    _z_log = False
    
    # TODO initally selected event must be within plot range!
    # Static widgets used for setting plot layout.
    x_field = 'cs1'
    x_bins = param.Tuple(default=(200, 0, 50), label='x binning (#bins, min max)')
    x_log = param.Boolean(default=False, label='Logarithmic x binning (np.logspace)')
    y_field = 'cs2'
    y_bins = param.Tuple(default=(200, 1000, 2500), label='y binning (#bins, min max)')
    y_log = param.Boolean(default=False, label='Logarithmic y binning (np.logspace)')
    clim = param.Tuple(default=(1, None), label='Colorbarlimit (min, max)')
    z_log = param.Boolean(default=True, label='Logarithmic colorbar')
    VALID_CONFIGS = ['x_field', 'x_bins', 'x_log', 'y_field', 'y_bins', 'y_log', 'clim', 'z_log']
    
    def __init__(self, 
                 events, 
                 default_config=immutabledict(),
                 exclude_fields=None,
                 _index_slider=None,
                 **params):
        for key, value in default_config.items():
            if not hasattr(self, key):
                raise ValueError(f'"{key}" is not a valid config. '
                                 f'Only the following attributes are recognized: {self.VALID_CONFIGS}.')
            setattr(self, key, value)
        
        self.exclude_fields = exclude_fields
        self.events = events
        self._create_widgets(self.x_field,
                             self.y_field,
                             _index_slider,
                            )
        super().__init__(**params)
        
        
    def interactive_event_stats(self):
        """An interactive plot which displays a histogram of the 
        supplied data. The user can manually change the axis and binning
        via the provided widgets. This plot is meant to be used as a 
        stand alone outside of the event display.
        """
        stream_hist, stream_marker = self._create_streams()
        hist = hv.DynamicMap(self._plot_hist_data, 
                             streams=stream_hist)
        selected_points = hv.DynamicMap(self._plot_selected_event, 
                                        streams=stream_marker)
        return pn.Column((hist*selected_points).opts(width=400), 
                         self._make_widget_panel(include_index_slider=True))

    def _create_streams(self):
        """Creates stream objects for the histogram and marker plot.
        """
        stream_hist = dict(
            x_field=self.x_field_widget, 
            y_field=self.y_field_widget,
            x_bins=self.param.x_bins, 
            y_bins=self.param.y_bins,
            clim=self.param.clim,
            x_log=self.param.x_log,
            y_log=self.param.y_log,
            z_log=self.param.z_log,
        )
        
        stream_marker=dict(
            x_field=self.x_field_widget, 
            y_field=self.y_field_widget,
            index=self.index_slider
        )
        
        return stream_hist, stream_marker
        
        
        
    def _create_widgets(self, 
                        x_field='cs1', 
                        y_field='cs2', 
                        _index_slider=None,
                       ):
        """Creates widgets for the x and y field as well as an slider
        for the event index if not supplied externally.
        """
        fields = list(self.events.dtype.names)
        if self.exclude_fields:
            if hasattr(self.exclude_fields, '__call__'):
                fields = self.exclude_fields(fields)
            elif isinstance(self.exclude_fields, (list, tuple)):
                fields = self._exclude_fields(fields, self.exclude_fields)
            else:
                raise ValueError('"exclude_fields" must be either a list or a function!')
                
        fields.sort()
        self.x_field_widget = pn.widgets.Select(
            value=x_field, 
            options=fields,
            name='X-field')
        
        self.y_field_widget = pn.widgets.Select(
            value=y_field, 
            options=fields,
            name='Y-field')
        
        if not _index_slider:
            self.index_slider = pn.widgets.IntSlider(start=0, 
                                                     end=len(self.events),
                                                     name='Event Index:'
                                                    )
        else:
            self.index_slider = _index_slider
    
    @staticmethod
    def _exclude_fields(fields, excluded_fields):
        """Function which drops certain fields from 
        """
        dtype = []
        for f in fields:
            if not f in excluded_fields:
                dtype.append(f)
        return dtype
        
    
    def _make_widget_panel(self, include_index_slider=False):
        """Put widgets used in the selection plot into a single panel
        which can be displayed. Adds index slider for stand alone plot.
        """
        widget_panel = pn.Row(
            pn.Column(self.x_field_widget,
                      self.param.x_bins, 
                      self.param.x_log,),
            pn.Column(self.y_field_widget, 
                      self.param.y_bins, 
                      self.param.y_log,),
            pn.Column(self.param.clim, 
                      self.param.z_log,),
            scroll=True,
        )
        
        if include_index_slider:
            widget_panel = pn.Column(self.index_slider, widget_panel)
        
        return widget_panel
        
    def _hist_data(self, x_field, y_field, xbins, ybins, x_log, y_log):
        """Function which checks if underlying data has to be rebinned.
        """
        _hist_has_not_changed = True
        _hist_has_not_changed &= x_field == self._x_field_name
        _hist_has_not_changed &= y_field == self._y_field_name
        _hist_has_not_changed &= xbins == self._x_bins
        _hist_has_not_changed &= x_log == self._x_log
        _hist_has_not_changed &= ybins == self._y_bins
        _hist_has_not_changed &= y_log == self._y_log
        

        if not _hist_has_not_changed:                    
            x_binning = self._get_binning(*xbins, x_log)
            y_binning = self._get_binning(*ybins, y_log)

            entries, x_edges, y_edges = np.histogram2d(
                self.events[x_field], 
                self.events[y_field],
                bins=(x_binning, 
                      y_binning,), 
            )

            self._x_edges = x_edges
            self._y_edges = y_edges
            self._entries = entries.T

            self._x_field_name = x_field
            self._y_field_name = y_field
            self._x_bins = xbins
            self._y_bins = ybins
            self._x_log = x_log
            self._y_log = y_log

    def _get_binning(self, n_bins, bin_min, bin_max, log):
        """Function which will return an array of bin edges.
        """
        if log:
            bin_min = max(10**-3, bin_min)
            bins = np.logspace(np.log10(bin_min),
                               np.log10(bin_max),
                               n_bins)
        else:
            bins = np.linspace(bin_min, 
                               bin_max, 
                               n_bins)
        return bins

    
    def _plot_hist_data(self, 
                  x_field: str,
                  y_field: str,
                  x_bins: tuple,
                  y_bins: tuple,
                  clim: tuple=(1, None),
                  x_log: bool=False,
                  y_log: bool=False,
                  z_log: bool=True,
                 ):
        """Creates histogram based on user specified settings.
        """
        self._hist_data(x_field,
                        y_field, 
                        x_bins, 
                        y_bins,
                        x_log,
                        y_log,
                       )
        hist = hv.QuadMesh(
            (self._x_edges, 
             self._y_edges,
             self._entries),
            [x_field, 
             y_field],
        )
        hist = hist.apply.opts(logz=z_log, 
                               logx=x_log,
                               logy=y_log,
                               clim=clim,
                               cmap='viridis', 
                               colorbar=True,
                               framewise=True,
                               clipping_colors={'min': 'white'},
                       )
        return hist
    
    
    def _plot_selected_event(self, 
                             x_field: str, 
                             y_field:str, 
                             index: int=0):
        """Function which highlights event with selected index in plot.
        """
        point_x = self.events[x_field][index]
        point_y = self.events[y_field][index]
        _is_nan = (np.isnan(point_x) or np.isnan(point_y))
        if not _is_nan:
            selected_points = hv.Points(
                (point_x, 
                 point_y),
                [self.x_field_widget.value,
                 self.y_field_widget.value
                ]
            ).opts(color='red', size=7, alpha=1)

            hline = hv.HLine(point_y).opts(color='red', alpha=0.4)
            vline = hv.VLine(point_x).opts(color='red', alpha=0.4)
        else:
            # In case one of the parameters is not defined 
            # plot invisilbe point
            selected_points = hv.Points(
                (0, 
                 0),
                [self.x_field_widget.value,
                 self.y_field_widget.value
                ]
            ).opts(color='k', alpha=0, size=7)

            hline = hv.HLine(0).opts(color='k', alpha=0)
            vline = hv.VLine(0).opts(color='k', alpha=0)
        
        return selected_points*hline*vline
