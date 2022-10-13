from immutabledict import immutabledict
import warnings
import panel as pn
import numpy as np
import holoviews as hv
import strax
from .holoviews_inspector import EventStats
from .holoviews_peak_data import PlotPeaksTPC
from .holoviews_pmt_array import PlotPMTArrayTPC

class InteractiveTPCEventDisplay():
    """Interactive event display for the TPC. Can be used for waveform
    watching/inspection. For waveform inspection the user can provide
    a subset of peak data which only belongs to the peaks of interest.
    In this case the inspector plot plots all events, but only the 
    selected events with peaks can be choosen with the event slider.
    
    :param events:
    :param peaks:
    :param selection:
    :param gains: Can be used to apply to_pe value to area_per_channel
        before plotting. PMTs with a to_pe value of zero are displayed
        in gray.
    """
    _current_event_index = -1
    
    def __init__(self, 
                 events, 
                 peaks,
                 gains=None,
                 plot_alt_peaks=True,
                ):
        self._events = events
        self._peaks = peaks
        self._plot_alt_peaks = plot_alt_peaks
        self._split_peaks_by_event()
        self._make_event_slider()
        self.gains = gains
        
    def event_display(self, 
                      inspector_config=immutabledict(),
                      exclude_fields=None,
                      inspection_mode=True,
                      plot_alt=True,
                      log_hit_pattern=True,
                      row_height=300,
                      row_width=900,
                     ):
        """Function which plots interactive event display.
        """
        
        plots = self._get_display_componnts(
            inspector_config=inspector_config,
            exclude_fields=exclude_fields,
            inspection_mode=inspection_mode,
            plot_alt=plot_alt,
            log_hit_pattern=log_hit_pattern,
        )             
        pattern, inspector_plot, inspector_widget, s2_peak, s1_peak, event_display = plots
        
        
        event_display.opts(height=row_height, width=row_width)
        row_setting = dict(height=row_height, width=row_width//3)
        s1_s2_pattern = pn.Row(s1_peak.opts(**row_setting), 
                               s2_peak.opts(**row_setting), 
                               pattern.opts(**row_setting))
        if inspection_mode:
            inspector_widget.width = row_width//2
            inspector_widget.height = row_height
            inspector_row = pn.Row(inspector_plot.opts(height=row_height, 
                                                       width=row_width//2),
                                   pn.Column(self.event_slider, inspector_widget)
                                  )
            return pn.Column(s1_s2_pattern, event_display, inspector_row)
        return pn.Column(self.event_slider, s1_s2_pattern, event_display)
        
    
    def _get_display_componnts(self, 
                               inspector_config=immutabledict(),
                               exclude_fields=None,
                               inspection_mode=True,
                               plot_alt=True,
                               log_hit_pattern=True,
                              ):
        """Function which creates all required plot and widget 
        components for the interactive event display. 
        """
        inspector_plot = None
        inspector_widget = None
        if inspection_mode:
            inspector_plot, inspector_widget = self._get_inspector_plot_and_widget(
                inspector_config, 
                exclude_fields
            )
        
        streams = {'event_index': self.event_slider}
        pattern = hv.DynamicMap(self._pmt_pattern_callback, 
                                streams=streams)
        
        s2_peak = hv.DynamicMap(
            self._s2_callback,
            streams=streams)
        s1_peak = hv.DynamicMap(
            self._s1_callback, 
            streams=streams)
        event_display = hv.DynamicMap(
            self._event_callback,
            streams=streams)
        
        return pattern, inspector_plot, inspector_widget, s2_peak, s1_peak, event_display
            
    def _get_inspector_plot_and_widget(self, inspector_config, exclude_fields):
        inspector = EventStats(self._events, 
                               inspector_config,
                               exclude_fields,
                               _index_slider=self.event_slider,
                              )
        stream_hist, stream_marker = inspector._create_streams()
        hist = hv.DynamicMap(inspector._plot_hist_data, 
                             streams=stream_hist)
        selected_points = hv.DynamicMap(inspector._plot_selected_event, 
                                        streams=stream_marker)

        inspector_plot = (hist*selected_points)
        inspector_widget = inspector._make_widget_panel(include_index_slider=False)
        return inspector_plot, inspector_widget
    
    def _selcet_current_event(self, event_index):
        """Function which selects current event. If event has been 
        already selected do nothing. This improves performance as we
        need four different callbacks.
        """
        if self._current_event_index == event_index:
            return
        
        self._selected_event = self._events[event_index]
        self._peaks_in_event = self._peaks[self._fully_contained_index == event_index]
        
        self._s2 = self._peaks_in_event[self._selected_event['s2_index']]
        self._s1 = self._peaks_in_event[self._selected_event['s1_index']]
        other_peaks_index = np.arange(len(self._peaks_in_event))
        mask_other_s2 = self._peaks_in_event['type'] == 2
        mask_other_s2 &= (other_peaks_index != self._selected_event['s2_index'])
        
        self._alt_s2 = None
        _can_plot_alt_s2 = (self._selected_event['alt_s2_index'] != -1) and self._plot_alt_peaks
        if _can_plot_alt_s2:
            self._alt_s2 = self._peaks_in_event[self._selected_event['alt_s2_index']]
            mask_other_s2 &= (other_peaks_index != self._selected_event['alt_s2_index'])
        
        self._alt_s1 = None
        _can_plot_alt_s1 = (self._selected_event['alt_s1_index'] != -1) and self._plot_alt_peaks
        if _can_plot_alt_s1:    
            self._alt_s1 = self._peaks_in_event[self._selected_event['alt_s1_index']]
        
        self._other_s2 = self._peaks_in_event[mask_other_s2]
        self._current_event_index = event_index
        
    
    def _pmt_pattern_callback(self, event_index, top_pmt_array=True, pattern_ops=immutabledict()):
        """Callback function for the PMT hitpattern. Selects main and 
        alt S2 position for a given event_index and creates the plot.
        """
        self._selcet_current_event(event_index)
        
        pmt_array = PlotPMTArrayTPC(
            gains=self.gains,
            top_pmt_array=top_pmt_array,
        )
        
        s2_plot = pmt_array.plot_pmt_array(self._s2, 
                                           label='S2', 
                                           logz=True, #TODO make me optional?
                                           **pattern_ops,
                                          )
        s2_point = hv.Points(
            (self._selected_event['x'], 
             self._selected_event['y']), 
            label='S2').opts(color='red', size=5)
        position_plot = s2_plot
        position_plot = (s2_plot * s2_point)
        
        if not self._alt_s2 is None:
            alt_s2_plot = pmt_array.plot_pmt_array(self._alt_s2, 
                                                   label='alt. S2', 
                                                   logz=True, #TODO make me optional?
                                                   **pattern_ops,
                                                  )
            alt_s2_point = hv.Points(
                (self._selected_event['alt_s2_x'], 
                 self._selected_event['alt_s2_y']), 
                label='alt. S2').opts(color='orange', size=5,) #Todo allow to cutsomize better...
            # Cannot use *= because changes plot order
            position_plot = (alt_s2_plot * alt_s2_point) * position_plot
        
        return position_plot.opts(legend_opts={"click_policy": "hide"})
   
    
    def _peak_callback(self, 
                       main_peak,
                       main_label,
                       alt_peak, 
                       alt_label,
                       time_in_µs=False,
                       opts_curve=immutabledict(),
                       opts_area=immutabledict(),
                       amplitude_prefix=''
                      ):
        """Call back which plots main/alt S1/S2 for display.
        """
        peak_ploter = PlotPeaksTPC()
        peak_plot = peak_ploter.plot_peak(main_peak,
                                          label=main_label,
                                          opts_curve=opts_curve,
                                          opts_area=opts_area,
                                          amplitude_prefix=amplitude_prefix,
                                          time_in_us=time_in_µs)
        if not alt_peak is None:
            opts_curve = self._add_legend_mute(opts_curve)
            opts_area = self._add_legend_mute(opts_area)
            alt_peak_plot = peak_ploter.plot_peak(alt_peak,
                                                  label=alt_label,
                                                  opts_curve=opts_curve,
                                                  opts_area=opts_area,
                                                  amplitude_prefix=amplitude_prefix,
                                                  time_in_us=time_in_µs)
        
            peak_plot *= alt_peak_plot
        return peak_plot
        
    
    def _s2_callback(self, 
                     event_index,
                     opts_peak_area=immutabledict(color='purple'),
                     opts_peak_curve=immutabledict(color='purple'),
                    ):
        """Call back which plots main/alt S2 for display.
        """
        self._selcet_current_event(event_index)
        return self._peak_callback(self._s2, 'S2', 
                                   self._alt_s2, 'alt. S2', 
                                   time_in_µs=True,
                                   opts_area=opts_peak_area,
                                   opts_curve=opts_peak_curve,
                                   amplitude_prefix='S2',
                                  ).opts(title='main/alt. S2')
    
    def _s1_callback(self, 
                     event_index,
                     opts_peak_area=immutabledict(color='orange'),
                     opts_peak_curve=immutabledict(color='orange'),
                    ):
        """Call back which plots main/alt S1 for display.
        """
        self._selcet_current_event(event_index)
        return self._peak_callback(self._s1, 'S1', 
                                   self._alt_s1, 'alt. S1', 
                                   time_in_µs=False,
                                   opts_area=opts_peak_area,
                                   opts_curve=opts_peak_curve,
                                   amplitude_prefix='S1',
                                  ).opts(title='main/alt. S1')
    
    def _event_callback(self, 
                        event_index,
                       ):
        """Callback function to plot the main display.
        """
        self._selcet_current_event(event_index)
        
        # Get event start time and extend a bit to the left:
        event_start = self._selected_event['time']
        event_end = self._selected_event['endtime']
        s1_start = self._selected_event['s1_time']
        rel_start = max(event_start, s1_start)
        peak_ploter = PlotPeaksTPC()                
        event_plot = []
        
        mask = self._peaks_in_event['type'] == 2
        self.n_s2 = np.sum(mask)
        if np.any(mask):
            plot_s2 = peak_ploter.plot_peaks(self._peaks_in_event[mask],
                                             label=f'S2 {event_index}',
                                             group_label=f'Event {event_index}',
                                             opts_curve={'color': 'purple', 'muted_alpha': 0,},
                                             opts_area={'color': 'purple', 'muted_alpha': 0,},
                                             time_in_us=True,
                                             time_prefix='Event',
                                             amplitude_prefix='Event',
                                             _relative_start_time=rel_start)
            event_plot.append(plot_s2)

        mask = self._peaks_in_event['type'] == 1
        self.n_s1 = np.sum(mask)
        if np.any(mask):
            plot_s1 = peak_ploter.plot_peaks(self._peaks_in_event[mask],
                                             label=f'S1 {event_index}',
                                             group_label=f'Event {event_index}',
                                             opts_curve={'color': 'orange', 'muted_alpha': 0,},
                                             opts_area={'color': 'orange', 'muted_alpha': 0,},
                                             time_in_us=True,
                                             time_prefix='Event',
                                             amplitude_prefix='Event',
                                             _relative_start_time=rel_start)
            event_plot.append(plot_s1)


        mask = self._peaks_in_event['type'] == 0
        self.n_s0 = np.sum(mask)
        if np.any(mask):
            plot_s0 = peak_ploter.plot_peaks(self._peaks_in_event[mask],
                                             label=f'S0 {event_index}',
                                             group_label=f'Event {event_index}',
                                             opts_curve={'color': 'gray',  'muted_alpha': 0,},
                                             opts_area={'color': 'gray', 'muted_alpha': 0,},
                                             time_in_us=True,
                                             time_prefix='Event',
                                             amplitude_prefix='Event',
                                             _relative_start_time=rel_start)
            event_plot.append(plot_s0)
        
        shaded_regions = self._get_alt_main_shading(self._selected_event, 
                                                    rel_start, in_µs=True)
        event_plot = hv.Overlay(event_plot + shaded_regions)
        return event_plot.opts(legend_opts={"click_policy": "hide",}, 
                               show_legend=True,
                               xlim=((event_start-rel_start)/10**3, 
                                     (event_end-rel_start)/10**3),
                               ylim=(0, None),
                              )
    
    
    def _get_alt_main_shading(self, event, start_time, in_µs=True):
        """Function which shades alt./main S1/S2 with 
        """
        time_scaler = 1
        if in_µs:
            time_scaler = 1000

        peak_times = dict()
        for peak in ['s1', 's2', 'alt_s1', 'alt_s2']:
            peak_start = event['_'.join((peak, 'time'))]
            peak_end = event['_'.join((peak, 'endtime'))]
            _has_peak = peak_start != -1
            if _has_peak:
                peak_times[peak] = ((peak_start - start_time)/time_scaler,
                                    (peak_end - start_time)/time_scaler,
                                    )
            else:
                peak_times[peak] = (-start_time, -start_time)

        shaded_regions = []
        for peak_type, (start, end) in peak_times.items():
            span = hv.VSpan(start, end).opts(color='gray', alpha=0.4)
            text = hv.Text(
                start, 0, rotation=90,
                text=peak_type, valign='bottom', halign='left').opts(text_alpha=1)

            # In case peak does not exists make label transparent. (We have 
            # to always make 4 labels as otherwise hv.Text breaks if we 
            # start with less peaks.)
            _no_peak = start == -start_time
            if _no_peak:
                text.opts(text_alpha=0)

            shaded_regions += [span, text]  

        return shaded_regions
        
    
    @staticmethod
    def _add_legend_mute(options):
        options = {k: v for k, v in options.items()}
        options['muted'] = True
        options['muted_alpha'] = 0
        return immutabledict(options)
        
    def _split_peaks_by_event(self):
        """Split provided peaks by event.
        """
        fci = strax.fully_contained_in(self._peaks, self._events)
        if np.any(fci == -1):
            warnings.warn('Found peaks which are not belonging to any '
                          'event. We are dropping these peaks for efficiency '
                          'purposes.')
        self._peaks = self._peaks[fci != -1]
        self._fully_contained_index = fci[fci != -1]
        self._events_with_peaks = list(np.unique(self._fully_contained_index))
    
    
    def _make_event_slider(self):
        """Creates slider for all events with peak data.
        """
        self.event_slider = pn.widgets.DiscreteSlider(
            name='Selected Event Index:',
            options=self._events_with_peaks, 
            value=self._events_with_peaks[0],
            value_throttled=True,
        )
