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
In addition to these dependencies, we also rely on an older version of `chemprop` ([https://github.com/chemprop/chemprop](https://github.com/chemprop/chemprop)). Clone this repository, checkout the required version, and install it as a package in the `mol2image` conda environment:
```
cd /path/to/chemprop
git checkout f9581c59483310b2eddae278b3507980c54249fa
pip install -e .
```
## Usage
### Generating Images
Download the pretrained model weights from [Google Drive](https://drive.google.com/drive/folders/1pSY62ylQj5YlHTrJ3CFZivim_vbvLLYy?usp=sharing) and place them in a directory called `pretrained`. To generate images corresponding to the molecules that were observed during training, run:
```
python generate.py --save-dir /path/to/results --val-metafile data/metadata/datasplit_gen_test_easy.csv
```
To generate images corresponding to the molecules that were held-out during training, run:
```
python generate.py --save-dir /path/to/results --val-metafile data/metadata/datasplit_gen_test_hard.csv
```
To generate images corresponding to the selected molecules for CellProfiler analysis, run:
```
python generate.py --save-dir /path/to/results --val-metafile data/metadata/datasplit_gen_test_easy_ext10.csv
python generate.py --save-dir /path/to/results --val-metafile data/metadata/datasplit_gen_test_hard_ext10_unique.csv
```
The generated and corresponding real images for the molecules will be saved to `/path/to/results/images`.

### CellProfiler Evaluation
To evaluate the generated images using `CellProfiler`, follow the installation instructions here: [https://github.com/CellProfiler/CellProfiler](https://github.com/CellProfiler/CellProfiler). Convert the generated .npz images to .png images (separate image for each channel) by running:
```
python convert_npz_to_png.py /path/to/results/images /path/to/results/png
```
Launch the CellProfiler GUI and open the pipeline from the file `mol2image.cpproj`. Then add the images in the directory `/path/to/results/png` to the pipeline and run.
