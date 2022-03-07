'''
# Single Electron Gain

Each time the anode is ramped up, there is an exponential relaxation period with a timescale of a day. The electron extraction efficiency and gain are affected, affecting S2 properties that make selection criteria invalid unless corrected. Here is provided the gain to make that correction.

The `ONLINE` version of the correction is used as the average SE gain we want to correct to within straxen. So for any given future run we won't correct for the SE gain in the `ONLINE` processing. Details can be found in the [straxen event processing](https://github.com/XENONnT/straxen/blob/master/straxen/plugins/event_processing.py#L597).

Each local version holds the SE gain at that time, extracted as a per run value.

A useful [note](https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:sr0_ramp_up_kr_se_study) by Jianyu.

'''