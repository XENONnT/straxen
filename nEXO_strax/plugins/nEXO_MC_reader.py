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
                            # entrysteps=1,
                            entrysteps='500 MB'
    )


@export
@strax.takes_config(
    strax.Option('input_dir', type=str, track=False,
                default='/home/brodsky3/nexo/mc_build/test2.root',
                 help="Directory where readers put data"),
    )
class MCreader(strax.Plugin):
    provides = ('photons','nest_hits')
    data_kind = {k: k for k in provides}
    depends_on = tuple()
    dtype = {'photons':
                (
                    ('energy', np.float, 'Photon energy (for wavelength)'),
                    ('type', np.int, 'photon origin type, 1=scint, 2=cherenkov'),
                    ('time', np.float, 'photon arrival time'),
                    ('endtime',np.float,'strax endtime,ignore'),
                    ('x', np.float, 'photon arrival x'),
                    ('y', np.float, 'photon arrival y'),
                    ('z', np.float, 'photon arrival z'),
                ),
            'nest_hits':
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

    rechunk_on_save = True

    def setup(self):
        self.data_iterator = get_from_path(self.config["input_dir"],
                                    ['OPEnergy','OPType','OPTime','OPX','OPY','OPZ',
                                     'NESTHitX','NESTHitY','NESTHitZ','NESTHitType','NESTHitT','NESTHitE','NESTHitNOP','NESTHitNTE'])

    def compute(self,chunk_i):

        g4_chunk = next(self.data_iterator)
        print(self.data_iterator, chunk_i)
        n_photons = len(g4_chunk[b'OPTime'].flatten())
        n_hits = len(g4_chunk[b'NESTHitT'].flatten())
        photon_records = np.zeros(n_photons, dtype=self.dtype['photons'])
        photon_records['energy'] = g4_chunk[b'OPEnergy'].flatten()

        hits = np.zeros(n_hits,dtype=self.dtype['nest_hits'])
        return dict(photons=photon_records, nest_hits = hits)

@export
class MCReader_test_consumer(strax.Plugin):
    depends_on = ['photons']
    provides = 'test_consumer'
    dtype = (('blah',np.float,'blahhh'),)
    def compute(self,chunk_i,photons):
        print(len(photons['energy']),chunk_i)
        return np.zeros(1,self.dtype)