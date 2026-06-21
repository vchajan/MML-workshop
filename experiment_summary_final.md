# MML Workshop – Image Colorization Final Summary

## Goal
The goal of the project was to improve automatic colorization of grayscale images. The original RGB/LAB regression baseline produced dull and averaged colours. The final direction therefore moved from direct a,b regression to a codebook-based colour prediction model.

## Dataset
Original dataset size: 4319 images.
A chroma-based dataset filter was applied to remove grayscale or strongly desaturated targets.

Filtered dataset:
- kept images: 3691
- removed low-chroma images: 628
- filter threshold: mean LAB chroma >= 8.0

This was necessary because some original training targets were actually black-and-white, which encouraged the model to produce desaturated outputs.

## Final model
Final selected model:

models/codebook_colorizer_filtered_v3/best.pt

Architecture:
- input: LAB L channel
- encoder: ResNet18
- decoder: U-Net style decoder
- output 1: classification over 256 learned LAB a,b colour-codebook bins
- output 2: residual a,b refinement

Final codebook:

models/codebook_filtered/ab_codebook_256_filtered.npz

## Why codebook instead of direct regression
Direct regression in LAB a,b space tends to average plausible colours. This causes gray, brown, greenish or desaturated outputs. The codebook approach predicts a distribution over learned colour bins and then refines the result with residual a,b prediction.

## Experiments
1. Original LAB Strong U-Net regression
   - technically worked
   - produced dull / averaged colours

2. Codebook model on unfiltered dataset
   - better colour semantics
   - still affected by grayscale targets in the dataset

3. Codebook model on filtered dataset
   - removed 628 low-chroma images
   - produced stronger and cleaner colours

4. v3 weighted colour experiment
   - increased pressure on colour-bin classification and saturated pixels
   - selected best checkpoint, not last checkpoint
   - final selected model: filtered_v3/best.pt

## Final interpretation
The final model is better than the original regression baseline. It produces more plausible colours and avoids some of the gray/green-brown averaging. It still cannot perfectly recover ambiguous colours from grayscale input, especially small red/orange objects, fantasy colour grading, or exact object colour when luminance does not contain enough information.

## Limitations
Image colorization from grayscale is inherently ambiguous. Multiple different RGB images can share almost the same L channel. Therefore, the model should be evaluated as plausible colourization, not exact reconstruction.

## Final demo output
outputs/final_v3_demo/final_v3_random_grid.png
