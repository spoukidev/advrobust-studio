"""AdvRobust Studio: adversarial attack & defense workbench for image classifiers.

All attacks operate on images in [0, 1] pixel space; models carry their own
normalization layer so the perturbation budget epsilon is always expressed in
pixel units (e.g. 8/255).
"""

__version__ = "0.1.0"
