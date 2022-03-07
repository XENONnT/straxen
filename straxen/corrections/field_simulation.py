'''
# Field simulation maps
Using finite element method (FEM) or boundary element method (BEM) analyses it is possible to simulate the electric field inside the TPC and simulate the impact that the field inhomogeneity has on the signal production. In case of freed electrons, some of their observed properties depends on the electric field along their full path (also called "path-weighted"): this is the case for time and position spread or for the drift speed. This can be evaluated by propagating electrons inside the TPC. The following maps have been evaluated so far:

 * electrostatic field map inside the cryostat (not only in the active volume);
 * survival probability within the active volume, meaning the probability of reaching the liquid-gas interface for an electron freed at a given position:
 * path-weighted drift speed;
 * path-weighted diffusion constants (ayimuthal, radial and along z).

In addition, notebooks on the field distortion coming from COMSOL simulations are included, as these are effectively simulation maps mapping the initial and final position.

See [overview on the wiki](https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:ftoschi:drift_diffusion_tpc_map_low_field).

'''

