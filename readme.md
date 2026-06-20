@"

# MML Workshop - Image Colorization

## Goal

The goal of this workshop is to colorize black-and-white photographs using an AI-based image colorization workflow.

## Project structure

- `notebooks/` - Jupyter notebook with the complete workflow
- `input_images/` - input black-and-white photos
- `outputs/` - generated colorized images
- `presentation/` - final 5-slide presentation
- `models/` - optional model files, not committed if large

## Workflow

1. Load black-and-white images.
2. Preprocess images.
3. Apply image colorization.
4. Save and compare results.
5. Summarize the method and results in a short presentation.

## Note

The project focuses on applying an existing colorization method and explaining the workflow, not on training a new model from scratch.
"@ | Set-Content "README.md" -Encoding UTF8
