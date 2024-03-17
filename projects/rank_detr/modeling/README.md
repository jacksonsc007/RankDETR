Branch name: reference points refinement of encoder (ref-refine)
# v1.0
The reference points in encoder layers are fixed in the original work. We propose reference points refinement following decoder.

# v1.1
we detach the gradient of sampling locations from previous layers, as the training process seems unstable.