import straxen, strax,ntauxfiles,sys
from time import time
import numpy as np
import matplotlib.cm as cm
from copy import deepcopy

sys.path.insert(0, r"/dali/lgrandi/guidam/GITHUB_XENON/CNN_S1_XYZ_DATADRIVEN/")
import plot_patterns

colormap = deepcopy(cm.YlGn)
colormap.set_under("w")

version_info = straxen.print_versions(modules=("straxen","strax","ntauxfiles"))

print(version_info)

st = straxen.contexts.xenonnt_online()

import sys
sys.path.insert(0, r"/dali/lgrandi/guidam/s1_posrec_pr1/straxen/straxen/plugins/events/")
import event_s1_position_cnn

st.register(event_s1_position_cnn.EventS1PositionCNN)

# run_id = '018221'
run_id = '025413'
stime = time()
recopos = st.get_array(run_id, "event_s1_position_cnn",_chunk_number=0)
keys = recopos.dtype.fields.keys()
print(keys)
etime = time()


print ("Totat Execution Time: ", round(etime-stime,3), " s")
print ("Total Number of Events: ", recopos.shape[0])
print ("Time per Event: ", round((etime-stime)/recopos.shape[0],6), " s")

# Patterns with Large Total Area Checking Plots
event_area_channel = st.get_array(run_id,'event_area_per_channel')

mask_area = event_area_channel['s1_area_per_channel'].sum(axis=1)>30000.
reco_pos_x = recopos['event_x_s1_cnn'][mask_area]
reco_pos_y = recopos['event_y_s1_cnn'][mask_area]
reco_pos_z = recopos['event_z_s1_cnn'][mask_area]
areas = event_area_channel['s1_area_per_channel'][mask_area]

print("================== DIMENSIONALITY CHECK ================== \n")

print("Shape X Reconstructed Vector : ",reco_pos_x.shape)
print("Shape Y Reconstructed Vector : ",reco_pos_y.shape)
print("Shape Z Reconstructed Vector : ",reco_pos_z.shape)
print("Shape Number of Patterns : ",areas.shape)

print("===========================================================  \n")

np.random.seed(14)

for i in range(2):
    # Random events for checking.
    ind=np.random.randint(0, areas.shape[0]) 
    pmtpos=straxen.pmt_positions()

    fig, axes = plot_patterns.plot_pattern(areas[ind], 
                                      pmtpositions=pmtpos,
                                      cmap = colormap,cbar_lim =[1e-10],cbar_scie=True)
    axes['ax_top'].plot(reco_pos_x[ind], reco_pos_y[ind], marker = "+",ms=30, color="orange",mew=6,label="S1 CNN Pos Rec\nz = "+str(round(reco_pos_z[ind],1))+ " cm")
    axes['ax_bottom'].plot(reco_pos_x[ind], reco_pos_y[ind], marker = "+",ms=30, color="orange",mew=6)
  
    axes['ax_top'].set_title("S1 - Top Array - Area per Channel (in PE)",fontsize=26)
    axes['ax_bottom'].set_title("S1 - Bottom Array - Area per Channel (in PE)",fontsize=26)

    axes['ax_top'].legend(fontsize=16)
    print("s1_area [PE] : ",areas[ind].sum())
    fig.savefig('/dali/lgrandi/guidam/s1_posrec_pr1/straxen/tests/plugins/temp_s1_posrec_test/plots_to_delate/example'+str(i)+'.png', bbox_inches='tight')

    # print("reconstructed position : ",recopos[['x_cnn_s1','y_cnn_s1','z_cnn_s1']][mask_area][ind])