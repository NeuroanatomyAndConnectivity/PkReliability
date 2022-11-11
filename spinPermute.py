#!/usr/bin/env python

from brainstat import tests
from brainspace.null_models import SpinPermutations
from brainspace.datasets import  load_marker, load_conte69
import pickle
import sys
seed=sys.argv[1]
print(f'using random seed {seed}')

# load the conte69 hemisphere surfaces and spheres
surf_lh, surf_rh = load_conte69()
sphere_lh, sphere_rh = load_conte69(as_sphere=True)
# Let's create some rotations
n_rand = 10
print(f'performing {n_rand} rotations')
sp = SpinPermutations(n_rep=n_rand, random_state=int(seed),unique=True)
sp.fit(sphere_lh, points_rh=sphere_rh)

with open(f'spins/spin_permutations000{seed}.pickle', 'wb') as handle:
    pickle.dump(sp, handle, protocol=pickle.HIGHEST_PROTOCOL)


