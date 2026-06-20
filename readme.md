# MML Workshop - Image Colorization

## Goal

The goal of this workshop is to train a simple custom image colorization model.

The model learns the mapping:

grayscale image -> color image

The task is not solved with a downloaded pretrained model. A small neural network is trained directly in the notebook.

## Project structure

- 
otebooks/colorization_workshop.ipynb - complete training and inference workflow
- data/dataset/ - local training image dataset, not committed to GitHub
- input_images/ - external black-and-white images for final teacher evaluation
- outputs/colorized/ - generated colorized examples from local test images
- outputs/comparisons/ - comparison grids: original / grayscale / predicted
- outputs/teacher_colorized/ - generated outputs for external teacher images
- outputs/teacher_comparisons/ - comparison grids for external teacher images
- models/ - trained model weights, local only unless explicitly required

## Method

1. Load color images from the local dataset.
2. Convert the images to grayscale.
3. Train a small CNN / U-Net-like encoder-decoder model.
4. Use the grayscale image as input.
5. Predict a three-channel RGB output.
6. Compare the generated image with the original color image.

## Final evaluation

For final evaluation, place black-and-white images into:

input_images/

Then run the inference section of the notebook. The colorized results will be saved into:

outputs/teacher_colorized/

and comparison images into:

outputs/teacher_comparisons/.

## Notes

The model is intentionally small so that it can be trained in a normal workshop environment. The output colors are approximate predictions, not guaranteed reconstructions of the original real-world colors.
