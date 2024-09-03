# MKID Detect

Inspired by [emccd_detect](https://github.com/roman-corgi/emccd_detect/tree/master/emccd_detect)
and the [MKID Pipeline](https://github.com/roman-corgi/emccd_detect/tree/master/emccd_detect),
for a given input flux map mkid_detect will return either an image or a simulated MKID photon list 
saved to an HDF5 file.

MKID data by nature can be quite memory intensive so caution should be taken to ensure memory
issues are not encountered.

## The MKIDDetect Class
There are two main ways to interact with MKIDDetect: The `sim_output` function and the `sim_output_image`
function. 

### `sim_output` 
will generate time and wavelength tagged photon lists similar to the outputs of 
an actual MKID detector. 

### `sim_output_image` 
Will output a single integrated image which, by
nature, marginalizes over all the timing information available to MKIDs but is much less time and 
memory intensive if just an image is desired.



## 



