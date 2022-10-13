import pandas as pd
import numpy as np
import straxen
from immutabledict import immutabledict
import holoviews as hv
import strax

class PlotPMTArrayTPC:
    """
    Class to plot TPC PMT array.
    
    :param gains: If a list of gain values are supplied PMTs which gain
        are set to zero are displayed in gray.
    :param top_pmt_array: Boolean whether top or bottom PMT array should
        be plotted.
    """
    def __init__(self, 
                 gains=None,
                 top_pmt_array=True,
                 opts=immutabledict(),
                ):
    
        self.is_top_pmt_array = top_pmt_array
        
        self._init_start_end_index()
        self._init_pmt_array(gains)
        self._init_common_options(opts)
        
    def _init_start_end_index(self):
        """Defines index-range of channels to be used for the 
        plot. Depends on whether the top or bottom array should be 
        plotted.
        """
        self.pmt_start = 0 if self.is_top_pmt_array else straxen.n_top_pmts+1
        self.pmt_end = straxen.n_top_pmts if self.is_top_pmt_array else straxen.n_tpc_pmts
        
        
    def _init_pmt_array(self, gains):
        """Initalized a dataframe which will contain the Top PMT 
        pattern array.
        """
        _pmts = pd.DataFrame()
        _pmts['channel'] = np.arange(0, straxen.n_tpc_pmts, 1)
        _pmts['x'] = straxen.pmt_positions().loc[:, 'x'].values
        _pmts['y'] = straxen.pmt_positions().loc[:, 'y'].values
        _pmts['gains'] = np.ones(straxen.n_tpc_pmts)
        if np.any(gains):
            if len(gains) != straxen.n_tpc_pmts:
                raise ValueError('Expected gains to have the same shape as number '
                                 'of PMTs in TPC!')
            _is_zero = gains == 0
            _pmts.loc[_is_zero, 'gains'] = np.nan
        self.pmts = _pmts.loc[self.pmt_start:self.pmt_end-1, :]
        
    def _init_common_options(self, opt_dict):
        """Creates dictionary with common options required by PMT array 
        plot.
        """
        if not isinstance(opt_dict, (dict, immutabledict)):
            raise ValueError('opts_dict must be a dictionary!')        
        
        if self.is_top_pmt_array:
            title = 'Top array top view'
        else:
            title = 'Bottom array top view'
            
        r = straxen.tpc_r
        r*=1.1
        default_settings = {'xlim': (-r, r),
                            'ylim': (-r, r),
                            'xlabel': 'x [cm]',
                            'ylabel': 'y [cm]',
                            'title': title
                           }
        self.settings = immutabledict({**default_settings, 
                                       **opt_dict})
    
    def _plot_tpc(self, **opts):
        """Function which plots the TPC as a simple circle.
        """
        tpc = straxen.holoviews_utils.plot_tpc_circle(straxen.tpc_r)
        tpc.opts(**opts)
        return tpc
    
    def _plot_top_array(self, peak_data, label, **opts):
        pmts = pd.DataFrame()
        pmts['area'] = peak_data['area_per_channel'][self.pmt_start:self.pmt_end]
        
        # Cops other static values: 
        # (we cannot use self.pmts as plot input as new plots will also change prev. 
        # plots due to same reference.)
        pmts['channel'] = self.pmts['channel']
        pmts['x'] = self.pmts['x']
        pmts['y'] = self.pmts['y']
        pmts['gains'] = self.pmts['gains']
        
        
        pmts['area'] /= pmts['gains'] 
        pmt_array = Circle(pmts, 
                           kdims = [hv.Dimension('x',label='x [cm]')],
                           vdims = [hv.Dimension('y', label='y [cm]'),
                                   hv.Dimension('area', label='Area [pe]'),
                                   hv.Dimension('channel', label='Channel')],
                           label=label,
                          )
        pmt_array.opts(color='area',
                       cmap='viridis',
                       line_color='black',
                       radius=3*2.54/2,
                       tools=['hover'],
                       clim=(10, None),
                       clipping_colors={'NaN': 'gray'},
                      )
        pmt_array.opts(**opts)
        return pmt_array
    
    def plot_pmt_array(self, peak, label='', **opts):
        """Function which plots the specified PMT array for the given 
        peak.
        
        :param peak: Peak for which PMT array should be plotted. Must 
            contain area per channel information.
        :param opts: Option which can be supplied to a holovies plot.
            E.g. you can use logz=True for a logarithmic colorscale or
            colorbar=True for adding a colorbar.
        :returns: Holoviews plot overlay.
        """
        # Allow to return empty plots required for event display if alt.
        if peak is None:
            return hv.Points(None)
        
        plot = self._plot_tpc() 
        plot *= self._plot_top_array(peak, label=label).opts(**self.settings, **opts)
        return plot
    
# Define function which plots points with radius defined in data space:
# Taken from: https://stackoverflow.com/questions/60361810/how-do-i-set-the-scatter-circle-radius-in-holoviews
import param
from holoviews.element.chart import Chart
from holoviews.plotting.bokeh import PointPlot


class Circle(Chart):
    group = param.String(default='Circle', constant=True)
    size = param.Integer()


class CirclePlot(PointPlot):
    _plot_methods = dict(single='circle', batched='circle')

    style_opts = ['radius' if so == 'size' else so for so in PointPlot.style_opts if so != 'marker']
hv.Store.register({Circle: CirclePlot}, 'bokeh')



# Test for straxen implementation:
def test_pmt_array_plot():
    """Test if PMT array can be plotted for the interactive 
    display.
    """
    dummy_gains = np.ones(494)
    dummy_gains[100:120] = 0
    
    dummy_peak = np.ones(1, dtype=strax.peak_dtype(n_channels=straxen.n_tpc_pmts))
    
    test = TPCPMTArray(gains=dummy_gains, top_pmt_array=True)
    test.plot_pmt_array(dummy_peak[0])
    
    test = TPCPMTArray(top_pmt_array=False)
    test.plot_pmt_array(dummy_peak[0])
