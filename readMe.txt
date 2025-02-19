The present framework composed of several pretrained super-resolution generative adversarial networks (SRGANs) achieves scale-oriented global field reconstructions from coarse flow fields in a zero-shot manner.

Sample codes for pretraining of super-resolution generative adversarial networks (SRGANs) based on snapshot samples from homogeneous and isotropic turbulence (HIT) are presented,
containing 1. main.py, 2. turSRGANs.py, 3. turNetwork.py and 4. utils.py. The file main.py is the main execution script, turSRGANs.py defines the training process, and turNetwork.py is used to construct the generator and discriminator structures of the SRGAN, as well as specifying the loss function calculations. The file utils.py contains the definitions of basic computational modules used within the SRGAN.  

The sample codes used for clustering the low-resolution global flow field into subdomains (5. Clustering.ipynb), as well as for pretraining the MLP models to estimate the Taylor microscales embedded in the low-resolution inputs (6. MLP.ipynb), are also provided.

Datasets:
All datasets used in the sample codes are available online. 
The pretraining data of homogenous and isotropic turbulence (HIT), as well as the testing data of turbulent boundary layer (TBL) and turbulent channel flow, are available in the Johns Hopkins Turbulence Databases (JHTDB) (http://turbulence.pha.jhu.edu/). 
The datasets of the climatological wind and ocean surface current data can be found at the public website (http://doi.org/10.22033/ESGF/CMIP6.3749/).

Requirements:
  -Python 3.x
  -keras
  -tensorflow 1.15
  -sklearn
  -numpy
  -pandas
