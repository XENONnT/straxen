import numba
import pandas
import straxen
import numpy as np
import strax
import pymongo
import typing
import matplotlib.pyplot as plt


@straxen.mini_analysis()
def daq_plot(context,
             figsize=(14, 15),
             lower_panel_height=6,
             group_by='link',
             vmin=None,
             vmax=None,
             **kwargs):
    """
    Plot with peak, records and records sorted by "link" or "ADC ID"
    (other items are also possible as long as it is in the channel map).
    """

    f, axes = plt.subplots(3, 1,
                           figsize=figsize,
                           gridspec_kw={'height_ratios': [1, 1, lower_panel_height]})

    # Panel 1, the peaks
    plt.sca(axes[0])
    plt.title('Peaks')
    context.plot_peaks(**kwargs,
                       single_figure=False)
    xlim = plt.xlim()
    plt.xticks(rotation=0)
    plt.grid('y')

    # Panel 2, the records where we keep the order of the records/channel number
    plt.sca(axes[1])
    plt.title('Records (by channel number)')
    context.plot_records_matrix(**kwargs,
                                vmin=vmin,
                                vmax=vmax,
                                single_figure=False)
    plt.xticks(rotation=0)
    plt.grid('x')
    plt.xlim(*xlim)

    # Use a grouping argument to group the channels by.
    plt.sca(axes[2])
    plt.title(f'Records (by {group_by})')
    context.plot_records_matrix(**kwargs,
                                vmin=vmin,
                                vmax=vmax,
                                group_by=group_by,
                                single_figure=False)
    plt.xlim(*xlim)
    plt.grid()


def _get_daq_config(
        context: strax.Context,
        run_id: str,
        config_name: str = 'daq_config',
        run_collection: typing.Optional[pymongo.collection.Collection] = None) -> dict:
    """
    Query the runs database for the config of the daq during this run.
    Either use the context of the runs collection.
    """
    if not context.storage[0].__class__.__name__ == 'RunDB' and run_collection is None:
        raise NotImplementedError('Only works with the runs-database')
    if run_collection is None:
        run_collection = context.storage[0].collection
    daq_doc = run_collection.find_one({"number": int(run_id)},
                                      projection={config_name: 1})
    if daq_doc is None or config_name not in daq_doc:
        raise ValueError(f'Requested {config_name} does not exist')
    return daq_doc[config_name]


def _board_to_host_link(daq_config: dict, board: int, add_crate=True) -> str:
    """Parse the daq-config to get the host, link and crate"""
    for bdoc in daq_config['boards']:
        try:
            if int(bdoc['board']) == board:
                res = f"{bdoc['host']}_link{bdoc['link']}"
                if add_crate:
                    res += f"_crate{bdoc['crate']}"
                return res
        except KeyError:
            raise ValueError(f'Invalid DAQ config {daq_config} or board {board}')
    # This happens if the board is not in the channel map which might
    # happen for very old runs.
    return 'unknown'


def _get_cable_map(name: str = 'xenonnt_cable_map.csv') -> pandas.DataFrame:
    """Download the cable map and return as a pandas dataframe"""
    down = straxen.MongoDownloader()
    cable_map = down.download_single(name)
    cable_map = pandas.read_csv(cable_map)
    return cable_map


def _group_channels_by_index(cable_map: pandas.DataFrame,
                             group_by: str = 'ADC ID',
                             ) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Parse the cable map, return the labels where each of the channels is
    mapped to as well as an array that can be used to map each of the
    channels maps to the labels.
    """
    idx = np.arange(straxen.n_tpc_pmts)
    idx_seen = 0
    labels = []
    for selection in np.unique(cable_map[group_by].values):
        selected_channels = cable_map[cable_map[group_by] == selection]['PMT Location']
        selected_channels = np.array(selected_channels)
        n_sel = len(selected_channels)

        idx[idx_seen:idx_seen + n_sel] = selected_channels
        labels += [selection] * n_sel
        idx_seen += n_sel
    return np.array(labels), idx


def group_by_daq(context, run_id, group_by: str):
    """From the channel map, get the mapping of channel number -> group by"""
    cable_map = _get_cable_map()
    if group_by == 'link':
        labels, idx = _group_channels_by_index(cable_map, group_by='ADC ID')
        daq_config = _get_daq_config(context, run_id)
        labels = [_board_to_host_link(daq_config, l) for l in labels]
        labels = np.array(labels)
        order = np.argsort(labels)
        return labels[order], idx[order]
    else:
        return _group_channels_by_index(cable_map, group_by=group_by)
