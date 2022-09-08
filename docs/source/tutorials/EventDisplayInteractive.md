# Interactive event display

In this short notebook we will explain how the interactive event display can be used and further customized. In the first section the basics are explained. Further, I show how the appearance of the event display can be customized. After the customization section I quickly explain how the record matrix addition works.


```python
import strax
import straxen
st = straxen.contexts.xenonnt_online(include_rucio_remote=True,
                                     download_heavy=True,
                                    )
```

In addition to our standard import we have to set up our jupyter notebook to show plots directly in the notebook:


```python
import bokeh.plotting as bklt
```

No let us get some events...


```python
# If data is not stored already download single chunk and make events
# for single chunk only:
if not st.is_stored('024399', 'event_basics'):
    st.get_array('024399', 'raw_records', _chunk_number=0)
    st.make('024399', 'event_basics', _chunk_number=0)
```


```python
events = st.get_array('024399', 'events', _chunk_number=0)
```

... and plot an arbitrary one:


```python
ith_event = 1
fig = st.event_display_interactive(run_id='024399', 
                                     time_range=(events[ith_event]['time'], 
                                                 events[ith_event]['endtime']),
                                  )
```

The parameter fig now stores the interactive plot as an instance of the bokeh.plotting class. Sounds complicated but showing the plot is actually quite simple. Before we plot the event it may be helpful to increase the cell width a bit. Although, I spend quite some time to get the auto-scaling of the plot working properly a too small width may lead to an overlap between legends and plots. 

For my 24 inch screen a width of 80% works fine. If you would like to go back to the default cell width just comment out the cell content below and execute the cell again.


```python
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))
```

To show the figure we can call:


```python
bklt.show(fig)
```

We now have an interactive event display with the following features:

* Mouseover which holds additional information for all shown Peaks, PMTs and S2 positions. In case you would like to deactivate the mouse over just deselect the speech bubble in the upper right corner of the plot's toolbar. 
* Interactive legends to select/deselect (semi-transparent) plots. E.g. if you push AS1 for alternative S1 in the upper right figure the alternative S1 will be shown.
* Moving, Box-Zoom and Reset tool: Additionally to the already mentioned feature the toolbar in the upper right corner offers a tool for moving (arrows) and zooming (magnifier). This can be helpful to zoom into smaller S1 S2 signals in the large event display in the bottom panel. To reset the zoom you can use arrows on a circle. 

You can also copy the plots to our wiki. For this purpose you can call...


```python
straxen.bokeh_to_wiki(fig)
```

... which converts the html image into a wiki friendly format. By default the out put is printed to the notebook so you can easily copy it to the wiki. You can also return the html code as a text file if you like. 

In case you would like to keep the data just as a normal html file you can do:


```python
bklt.save(fig, 'example.html')
```

but please be aware that in this case you have to change the first lines of the file into  
```html
<!DOCTYPE html>
<html lang="en">
```
into
```html
<html>
```
if you want to copy the output to the wiki.

In case you would like to store the event display as a png or svg file please follow this [guide](https://docs.bokeh.org/en/latest/docs/user_guide/export.html).

### Manipulating the display appearance:

For talks or a thesis it might be necessary to modify some of the fonts styles or size. Further, it may be needed to reduce the amount of shown information. For this purpose I added some additional options: 

* only_main_peaks: If True only the main S1/S2 are plotted in the upper row of plots (detailed peak view and PMT arrays).
* only_peak_detail_in_wf: If True plots only the peaks shown in the upper row into the waveform plot (e.g. if only_main_peaks == False only plots main Peaks).
* bottom_pmt_array: If False removes bottom PMT array from the upper row.
* plot_all_pmts: if true PMTs which are switched off are displayed in the same colors as the other PMTs. By default (False) these PMTs are colored gray.
* colors: Changes the colors of the different peak types. The colors are consistently used throughout all plots. The colors have to be specified in the same order as the peak types (Unknown, S1, S2). Support are bokeh.colors e.g. words, rgba values or hex values.


```python
ith_event = 1
fig = st.event_display_interactive(run_id='024399', 
                                     time_range=(events[ith_event]['time'], 
                                                 events[ith_event]['endtime']),
                                   only_main_peaks=True,
                                   only_peak_detail_in_wf=True,
                                   bottom_pmt_array=False,
                                   plot_all_pmts=False,
                                   colors=('orange', 'magenta', 'black')
                            )
```


```python
bklt.show(fig)
```

### How to customize the display even further? 

The entire display is build up by many individual figure objects. To customize certain properties we have to add them to the corresponding figure. To get the individual components one can use `fig.children`...


```python
fig.children
```

... which returns us a list of two objects a DIV object which represents the plot title and a Column object. As you can see already know the event display is actually a very nested thing. If we unpack the column object even further we get ...


```python
fig.children[1].children
```

... a toolbar and a GridBox. The toolbar represents the toolbar on top of our plots. The Grid box is another nested structure of the upper plot row and the lower waveform. To change now the properties of something specifically we have to get the corresponding object and adjust or add the desired option. Let us for example change the headline color to green. First we have to get the DIV object which uses HTML5 syntax for the styling:


```python
div = fig.children[0]
```

Lets add the color green:


```python
div.style['color'] = 'green'
```

Before we plot let us also remove the toolbar:


```python
tool_box = fig.children[1].children[0]
tool_box.visible=False
```


```python
bklt.show(fig)
```

# Record matrix: 

The regular event display can be extended by an additional record matrix which is helpful especially for peak building and splitting analyses. To plot the record matrix you have to simply set ` plot_record_matrix=True`. The event display will then automatically check if the specified raw_data is available. The record matrix is build on records, in case only `raw_records` are available the event display will warn you and build the required `records` on the fly.    


```python
if not st.is_stored('024399', 'records'):
    st.make('024399', 'records', _chunk_number=0)
```


```python
ith_event = 2
fig = st.event_display_interactive(run_id='024399', 
                                     time_range=(events[ith_event]['time'], 
                                                 events[ith_event]['endtime']),
                                   plot_record_matrix=True,
                                   plot_records_threshold=10,
                            )
```

Please note that in case of ` plot_record_matrix=True` you do not have to call `bklt.show()` as the display library has now changed to panel. Yes I know.... super annoying, but belief me implementing this was even more frustrating....   

The record matrix is displayed as as blue squares which are resized depending on the zoom level. Each square is allocated at the center time of the corresponding records. You can change the zoom level by hovering with the mouse over the central peak waveform plot and scrolling up and down with your mouse wheel. As an alternative you can also use the box zoom feature in the same plot. 
Once you zoomed in far enough, such that the x_range spans only 10 µs (default can be adjusted), the event display will plot all records inside the currently shown window as lines. The line color represents hereby the area of the individual records. If you zoom out again the lines vanish. 


```python
fig
```

The re-plotting of the record array requires an active python kernel in the background. This means that the interactivity of this plot gets lost as soon as it is exported as an HTML file.

# Event selection:

Beside an interactive event display we also have now an interactive cut selection tool. To use the tool we have to first import get some data:


```python
peaks = st.get_array('024399', 'peak_basics', _chunk_number=0)
```

Now we can import the tool...


```python
from straxen.analyses.bokeh_waveform_plot import DataSelectionHist
```

... and initialize it. Please note, that you have to put the class object name as a string into the first function argument. Otherwise the JavaScript call back for the data selection wont work.


```python
ds = DataSelectionHist('ds')
```

Now we can plot any data e.g. some peak properties...


```python
f = ds.histogram2d(peaks, 
                   peaks['area'], 
                   peaks['range_50p_area'], 
                   bins=50, 
                   hist_range=((0, 200), (0, 2000)), 
                   log_color_scale=True, clim=(10, None), undeflow_color='white')
```

... and with show we can render the plot. 

Please note that the current implementation is limited to a squared binning (in pixels not in data space as shown). The plot has the typical mousover feature showing the bin position and number of entries. You can deactivate this feature by clicking on speech bubble in the toolbar. 

However, the main feature is not the mousover but the dynamic data selection. Currently, two tools are supported for the data selection a box cut and a lasso cut allowing for arbitrary selections. By default the lasso tool is active. In general the box tool has a better performance since the computation is easier.

If you would like to make multiple selections at once you can press shift to add the selected areas up. 


```python
import bokeh
bokeh.io.output_notebook()
```


```python
bklt.show(f)
```

Once you are satisfied with your selection you can return the corresponding peaks with...


```python
ps = ds.get_back_selected_items()
```

... which stores the information is ps. Let us now finally make a control plot to see if the selection work. For this plot we use a finder binning:


```python
import matplotlib.pyplot as plt
import matplotlib
```


```python
plt.figure(dpi=150)
plt.hist2d(peaks['area'], peaks['range_50p_area'], 
           bins=(200, 200), 
           range=((0, 200), (0, 2000)), norm=matplotlib.colors.LogNorm(), cmap='inferno')
plt.hist2d(ps['area'], ps['range_50p_area'], 
           bins=(200, 200), 
           range=((0, 200), (0, 2000)), norm=matplotlib.colors.LogNorm(), cmap='viridis')
plt.xlabel('Area [pe]')
plt.ylabel('50 % Width [ns]')
plt.show()
```
