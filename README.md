Needle
======

This repository contains Python scripts for finding the orientation of an
elongated object in an image. The author uses it for tracking nanowires
in videomicroscopy data.

The package includes:
* generic image preparation tools shared by multiple tracking methods
* several different methods for tracking wire orientation which can be called
using a uniform interface, making it easy to switch between methods
  * finding the principal ("inertial") axes
  * fitting an elispse to the wire's shadow
  * finding the wire's Gaussian center at each row of pixels
  and fitting those centers to a line
* animated annotated plots, rendered in IPython notebooks as HTML5 videos,
  for directly viewing results
* a simple test suite to ensure code stability and correctness
