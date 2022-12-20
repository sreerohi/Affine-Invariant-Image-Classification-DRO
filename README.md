# Affine-Invariant-Image-Classification-DRO

It is well know that Convolutional Neural Networks are not robust to adversarial
transformations of domain. In this project we focus on affine image transformations.
Athough many methods have been proposed to achieve affine invariance, they suffer
from a resource barrier: given an d-dimensional family of transformations, data and
resources exponential in d are required for learning [1]. In this project, we propose
a novel optimization based approach to achieve invariance to affine transformations
that avoids this resource barrier and also requires minimal modification to the CNN
architecture. Although results of this approach are yet to be promising, this project
can lay the foundation stone for adding robustness to more complicated image
transformation families beyond affine transformations.

Citation:

@software{Pulipaka_Affine_Invariant_Image_2022,
author = {Pulipaka, Sree Rohith},
month = {12},
title = {{Affine Invariant Image Classification through DRO}},
version = {1.0.0},
year = {2022}
}

References:

[1] https://github.com/sdbuch/refine

[2] https://zablo.net/
