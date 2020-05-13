import strax
export, __all__ = strax.exporter()
import numpy as np

'''
The plan:
for each hit, create channels. Implicit binning

new data_kind: channel WF.
depends on hits with hit_channels
assemble hit_channels into channel WFs
implement global clock 
'''
# @export
# @strax.takes_config(
#     # strax.Option('local_channel_map', default='/home/brodsky3/nexo/nexo-offline/data/localChannelsMap_6mm.txt',help="local channel map file location"),
#     # strax.Option('tile_map', default='/home/brodsky3/nexo/nexo-offline/data/tilesMap_6mm.txt',help="local channel map file location"),
#     strax.Option('PCD_spacing_xy', default=.1,help="PCD spacing in mm"),
#     strax.Option('hit_nPCDs', default=50,help="number of PCDs"),
# )
# class hit_channels(strax.Plugin):
#     depends_on = 'nest_hits'
#     data_kind='nest_hits'
#     provides = 'hit_channels'
#     def infer_dtype(self):
#         return [(('Hit PCD expectation','hit_PCD_expectation'),np.int32,self.config['hit_nPCDs']),
#                 (('Hit PCD quanta', 'hit_PCD_quanta'), np.int32, self.config['hit_nPCDs']),
#                 ]
#     save_when = strax.SaveWhen.TARGET #only save if it's the final target
#
#     def setup(self):
#         pass
#
#     def compute(self, nest_hits):
#         for hit in nest_hits:
#
#
#
#         return result

@export
class Thermalelectrons(strax.Plugin):
    depends_on = 'nest_hits'
    data_kind = 'thermalelectrons'
    provides = 'thermalelectrons_original'
    dtype = [
        ('x',np.float,'original x position'),
        ('y', np.float, 'original y position'),
        ('z', np.float, 'original z position'),
        ('time', np.float64, 'original time'),
        ('endtime', np.float64, 'strax endtime, ignore')

    ]
    save_when = strax.SaveWhen.TARGET
    def compute(self,nest_hits):
        result = np.zeros(nest_hits['n_electrons'].sum(),dtype=self.dtype)
        for field in self.dtype.names:
            result[field]=np.repeat(nest_hits[field],nest_hits['n_electrons'])
        return result

@export
@strax.takes_config(
            strax.Option(f'anode_z', type=float, track=True, default=400, help="Anode z position, mm"),
        )
@strax.takes_config(
            strax.Option(f'drift_speed', type=float, track=True, default=.001, help="drift speed, mm/ns"),
        )
class Thermalelectrons_drift(strax.Plugin):
    depends_on = 'thermalelectrons_original'
    data_kind = 'thermalelectrons'
    provides = 'thermalelectrons_drift'
    dtype = [
        ('x_drift',np.float,'drift x position'),
        ('y_drift', np.float, 'drift y position'),
        ('z_drift', np.float, 'drift z position'),
        ('drifttime', np.float, 'drift time'),
        # ('endtime', np.float64, 'strax endtime, ignore')

    ]+strax.time_fields
    anode_z = 400
    save_when = strax.SaveWhen.TARGET
    def compute(self,thermalelectrons):
        result = np.zeros(len(thermalelectrons),dtype=self.dtype)
        drift_time = self.config['drift_speed']*(self.config['anode_z'] - thermalelectrons['z'])
        result['endtime']= thermalelectrons['endtime'] + drift_time
        result['time'] = thermalelectrons['time']
        return result

@export
class Test_consumer(strax.Plugin):
    depends_on = ['thermalelectrons_original','thermalelectrons_drift']
    provides = 'test_consumer2'
    dtype = strax.time_fields
    def compute(self,chunk_i,thermalelectrons):
        print(thermalelectrons['time'],thermalelectrons['endtime'],chunk_i)
        return np.zeros(1,self.dtype)