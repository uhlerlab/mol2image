# Mol2Image: Improved Conditional Flow Models for Molecule to Image Synthesis

This is the accompanying code for the paper, "Mol2Image: Improved Conditional Flow Models for Molecule to Image Synthesis" ([CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Mol2Image_Improved_Conditional_Flow_Models_for_Molecule_to_Image_Synthesis_CVPR_2021_paper.pdf)).

## Dataset

We use the subset of pre-processed images from the "Cell Painting Assay Dataset" provided by [Hofmarcher et al., (2019)](https://github.com/ml-jku/hti-cnn). Their dataset can be directly accessed here: [https://ml.jku.at/software/cellpainting/dataset](https://ml.jku.at/software/cellpainting/dataset). Download and unzip the images (in .npz format), and place them in a directory called `data/images`.

For the full cell painting dataset, see [https://github.com/gigascience/paper-bray2017](https://github.com/gigascience/paper-bray2017).

## Dependencies

Python dependencies can be installed via `conda` from the `environment.yml` file:
```
conda env create -f environment.yml
conda activate mol2image
```
In addition to these dependencies, we also rely on an older version of `chemprop` forked [here](https://github.com/uhlerlab/chemprop). Clone this repository and install it as a package in the `mol2image` conda environment:
```
cd /path/to/chemprop
pip install -e .
```
