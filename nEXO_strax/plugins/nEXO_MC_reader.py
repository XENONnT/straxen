import uproot
import strax
import numpy as np
from glob import glob
export, __all__ = strax.exporter()

@export
def get_tree(file):
    file_contents = uproot.open(file)
    all_ttrees = dict(file_contents.allitems(filterclass=lambda cls: issubclass(cls, uproot.tree.TTreeMethods)))
    tree = all_ttrees[b'nEXOevents;1']
    return tree

@export
def get_from_tree(tree,arrays):
    return tree.lazyarrays(arrays,entrysteps='10 MB',cache=uproot.cache.ArrayCache('500 MB'))

@export
def get_from_path(path,arrays):
    all_files = glob(path,recursive=True)
    return uproot.iterate(all_files,
                            b'nEXOevents',
                            arrays,
                            entrysteps=3,
                            # entrysteps='500 MB'
    )

@export
class MCreader(strax.Plugin):
    time: float
    depends_on = tuple()
    dtype_original = {f'photons': #pre-configuration for individual source
        (
            ('energy', np.float, 'Photon energy (for wavelength)'),
            ('type', np.int, 'photon origin type, 1=scint, 2=cherenkov'),
            ('time', np.float, 'photon arrival time'),
            ('endtime', np.float, 'strax endtime,ignore'),
            ('x', np.float, 'photon arrival x'),
            ('y', np.float, 'photon arrival y'),
            ('z', np.float, 'photon arrival z'),
        ),
        f'nest_hits':
            (
                ('x', np.float, 'hit x'),
                ('y', np.float, 'hit y'),
                ('z', np.float, 'hit z'),
                ('type', np.int, 'NEST interaction type'),
                ('time', np.float, 'hit time'),
                ('endtime', np.float, 'strax endtime,ignore'),
                ('energy', np.float, 'energy deposit in this hit'),
                ('n_photons', np.int, 'number of photons produced'),
                ('n_electrons', np.int, 'number of electrons produced'),
            ),

    }
    rechunk_on_save = False


    def infer_dtype(self):
        return {f'photons_{self.sourcename}': self.dtype_original['photons'],
                 f'nest_hits_{self.sourcename}': self.dtype_original['nest_hits']
                 }

    def setup(self):
        self.data_iterator = get_from_path(self.config[f"input_dir_{self.sourcename}"],
                                    ['OPEnergy','OPType','OPTime','OPX','OPY','OPZ',
                                     'NESTHitX','NESTHitY','NESTHitZ','NESTHitType','NESTHitT','NESTHitE','NESTHitNOP','NESTHitNTE'])
        self.time = self.config[f'start_time_{self.sourcename}']

    def compute(self,chunk_i):
        g4_chunk = next(self.data_iterator)
        evttimes = np.random.exponential(1/self.config[f'rate_{self.sourcename}'],(g4_chunk[b'OPTime'].size,)).cumsum()+self.time
        self.time = evttimes[-1] #increment for next round
        n_photons = len(g4_chunk[b'OPTime'].flatten())
        n_hits = len(g4_chunk[b'NESTHitT'].flatten())
        photon_records = np.zeros(n_photons, dtype=self.dtype[f'photons_{self.sourcename}'])
        photon_records['energy'] = (g4_chunk[b'OPEnergy']).flatten()
        photon_records['type'] = g4_chunk[b'OPType'].flatten()
        photon_records['time'] = (g4_chunk[b'OPTime']-g4_chunk[b'OPTime']+evttimes).flatten()
        photon_records['endtime'] = photon_records['time']
        photon_records['x'] = g4_chunk[b'OPX'].flatten()
        photon_records['y'] = g4_chunk[b'OPY'].flatten()
        photon_records['z'] = g4_chunk[b'OPZ'].flatten()

        hits = np.zeros(n_hits,dtype=self.dtype[f'nest_hits_{self.sourcename}'])
        hits['x'] = g4_chunk[b'NESTHitX'].flatten()
        hits['y'] = g4_chunk[b'NESTHitY'].flatten()
        hits['z'] = g4_chunk[b'NESTHitZ'].flatten()
        hits['type'] = g4_chunk[b'NESTHitType'].flatten()
        hits['time'] = (g4_chunk[b'NESTHitT']+evttimes).flatten()
        hits['endtime'] = hits['time']
        hits['energy'] = g4_chunk[b'NESTHitE'].flatten()
        hits['n_photons'] = g4_chunk[b'NESTHitNOP'].flatten()
        hits['n_electrons'] = g4_chunk[b'NESTHitNTE'].flatten()

        print(f"Loaded source from {self.config[f'input_dir_{self.sourcename}']} chunk {chunk_i} time {self.time} name {self.sourcename}")
        return {f'photons_{self.sourcename}': photon_records, f'nest_hits_{self.sourcename}': hits}

import numba
@numba.jit(nopython=True, nogil=True, cache=True)
def sort_by_time(x):
    """Sort pulses by time
    """
    if len(x) == 0:
        # Nothing to do, and .min() on empty array doesn't work, so:
        return x
    sort_key = (x['time'] - x['time'].min())
    sort_i = np.argsort(sort_key)
    return x[sort_i]

@export
class MCreader_factory(object):
    names = []
    source_plugins={}
    def make_MCreader(self, name : str, path:str, rate:float , start_time:float):
        self.names.append(name)
        @strax.takes_config(
            strax.Option(f'input_dir_{name}', type=str, track=True,
                         default=path,
                         help="Directory where readers put data"),
            strax.Option(f'rate_{name}', type=float, track=True,
                         default=rate,
                         help="rate [Hz] of this source"),
            strax.Option(f'start_time_{name}', type=float, track=True,
                         default=start_time,
                         help="start time [s] of this source"),
        )
        class newMCreader(MCreader):
            sourcename = name
            provides = [f'photons_{sourcename}', f'nest_hits_{sourcename}']
            data_kind = {k: k for k in provides}

        newMCreader.__name__ = f'MCreader_{name}'
        self.source_plugins[name]=newMCreader
        return newMCreader

    def make_MCmerger(self):
        assert len(self.names)>0
        class MCmerger(strax.Plugin):
            depends_on = [f'photons_{sourcename}' for sourcename in self.names] + [f'nest_hits_{sourcename}' for sourcename in self.names]
            provides = ['photons','nest_hits']
            data_kind = {k: k for k in provides}
            dtype = MCreader.dtype_original
            rechunk_on_save = False
            def compute(self, **kwargs):
                output_dict = {}
                for dtype_name, dtype_val in self.dtype.items():
                    type_args = [arg for arg in kwargs.items() if dtype_name in arg[0] ]
                    input_arrays = np.concatenate([arg[1] for arg in type_args])
                    output = sort_by_time(input_arrays)
                    output_dict[dtype_name] = output
                    print(f'merged time {output["time"] }')

                return output_dict
        return MCmerger










@export
class MCReader_test_consumer(strax.Plugin):
    depends_on = ['photons']
    provides = 'test_consumer'
    dtype = (('blah',np.float,'blahhh'),)
    def compute(self,chunk_i,photons):
        print(photons['time'],chunk_i)
        return np.zeros(1,self.dtype)

