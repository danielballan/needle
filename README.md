Track Wire
==========

This repository contains Python scripts for finding the orientation of a wire 
in an image.

It includes
* generic preprocessing tools used by multiple methods
* several different methods for tracking wire orientation which can be called
using a uniform interface, making it easy to switch between methods
  * fitting an elispse to the wire's shadow
  * finding the wire's Gaussian center at each row of pixels
  and fitting those centers to a line
  * finding the principal ("inertial") axes
* a simple test suite to ensure code stability and correctness
