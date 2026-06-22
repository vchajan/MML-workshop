# Final Image Colorization Training Summary

## Final selected model

The final model was trained from random initialization without pretrained weights.

Checkpoint:

models/codebook_colorizer_scratch_v2/best.pt

## Training setup

- Dataset: data/dataset_filtered
- Images: 3691
- Train split: 3322
- Validation split: 369
- Image size: 256 x 256
- Batch size: 4
- Epochs requested: 40
- Final selected checkpoint: epoch 22
- Best validation loss: 15.862982032119588
- GPU: NVIDIA GeForce RTX 2070
- AMP: enabled
- Encoder initialization: random
- Pretrained weights: False

## Model idea

The project started with direct RGB prediction. This baseline worked technically, but produced dull and averaged colours.

The final approach uses LAB colour space:

- input: L channel
- target: a,b colour channels
- 256 learned AB colour bins
- classification over colour bins
- residual a,b refinement

## Final training command

python scripts/train_codebook_colorizer.py --data data/dataset_filtered --codebook models/codebook_filtered/ab_codebook_256_filtered.npz --epochs 40 --batch-size 4 --image-size 256 --num-workers 2 --amp --temperature 0.28 --lambda-ce 2.0 --lambda-ab 6.0 --chroma-weight 8.0 --model-dir models/codebook_colorizer_scratch_v2 --sample-dir outputs/codebook_samples_scratch_v2

No pretrained encoder flag was used.

## Final demo

docs/final_scratch_v2_random_grid.png
