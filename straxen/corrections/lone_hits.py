'''
# Correction lone_hits evolution

As studied and discussed in detail in [this note](https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:dsg:pmt:gains:stability_sr0:lonehits), we could observe a temporal change of the `lone_hits` areas even after correction with the [PMT gains from LED calibrations](https://github.com/XENONnT/corrections/tree/lone_hits_correction/XENONnT/pmt_gains) in XENONnT SR0. As this effect also seems to impact for example the light yield evolution and might be the cause for discrepancies in the Doke-plot, we decided to introduce an emprical correction factor. For simplicity, this correction factor is, for now, the same for all PMTs and based on a piecewise function (23 intervals, constant / linear / exponential), fitted to the median lone_hits evolution for SR0 PMTs and normalized to the stable background data (set to correction factor one). 

The parameters of this correction function are given in the `Lone_hits_correction_SR0.pkl/csv/h5` files (different formats provided in case of incompatibilities). Help and examples on how to apply this correction is given in `Lone_hits_correction_SR0_mockup.ipynb`.


'''