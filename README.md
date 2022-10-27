<h1 align="center"><img src=".github/logo.svg"
  width=45px>
  MidasTouch
</h1>
<h2 align="center">
  Monte-Carlo inference over distributions across sliding touch
</h2>


<div align="center">
  <a href="https://www.cs.cmu.edu/~sudhars1/">Sudharshan Suresh</a> &nbsp;•&nbsp;
  <a href="https://si-lynnn.github.io/">Zilin Si</a> &nbsp;•&nbsp;
  <a href="https://scholar.google.com/citations?user=8orqBsYAAAAJ&hl=en">Stuart Anderson</a> &nbsp;•&nbsp;
  <a href="https://www.cs.cmu.edu/~kaess/">Michael Kaess</a> &nbsp;•&nbsp;
  <a href="https://www.mustafamukadam.com/">Mustafa Mukadam</a>
  <br/>
  6th Annual Conference on Robot Learning (<a href="https://corl2022.org/">CoRL</a>) 2022
</div>

<h4 align="center">
  <a href="https://suddhu.github.io/midastouch-tactile/"><b>Website</b></a> &nbsp;•&nbsp;
  <a href="https://arxiv.org/abs/2210.14210"><b>Paper</b></a> &nbsp;•&nbsp; 
  <a href="https://youtu.be/MkudcePyBJI"><b>Presentation</b></a> &nbsp;•&nbsp; 
  <a href="https://github.com/rpl-cmu/YCB-Slide"><b>YCB-Slide</b></a>
</h4>

<div align="center">
<b>TL;DR</b>: We track the pose distribution of a robot finger on an <br>
object's surface using geometry captured by a tactile sensor
<br> <br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) &nbsp; [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img height="20" src=".github/meta_ai.jpeg" alt="Meta-AI" />  &nbsp;&nbsp; <img height="20" src=".github/rpl.png" alt="rpl" />
</div>

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg


MidasTouch performs online global localization of a vision-based touch sensor on an object surface during sliding interactions.  For details and further results, refer to our <a href="https://suddhu.github.io/midastouch-tactile/">website</a> and <a href="https://openreview.net/forum?id=JWROnOf4w-K">paper</a>.

<div align="center">
  <img src=".github/021_bleach_cleanser.gif"
  width="80%">
</div>



## Setup

### 1. Clone repository

```bash
git clone git@github.com:facebookresearch/MidasTouch.git
git submodule update --init --recursive
```
### 2. Download  <a href="https://github.com/rpl-cmu/YCB-Slide">YCB-Slide</a> dataset
```bash
cd YCB-Slide 
chmod +x download_dataset.sh && ./download_dataset.sh
cd ..
```
### 3. Download weights/codebooks
```bash
chmod +x download_assets.sh && ./download_assets.sh
```
### 4. Setup midastouch conda env
```bash
sudo apt install build-essential python3-dev libopenblas-dev
conda env create -f environment.yml 
conda activate midastouch
pip install -e .
```
### 5. Install PyTorch and the MinkowskiEngine

&nbsp; &nbsp; &nbsp; Follow [the conda instructions](https://github.com/NVIDIA/MinkowskiEngine#anaconda) from the NVIDIA MinkowskiEngine webpage 



## Run MidasTouch

Run interactive filtering experiments with our YCB-Slide data from both the simulated and real-world tactile interactions. 

<div align="center">
  <img src=".github/interactive_filter.gif"
  width="60%">
</div>


### TACTO simulation trajectories
```python
python midastouch/filter/filter.py expt=ycb # default: 004_sugar_box log 0
python midastouch/filter/filter.py expt.obj_model=035_power_drill expt.log_id=3 # 035_power_drill log 3
python midastouch/filter/filter.py expt.off_screen=True   # disable visualization
python midastouch/filter/filter.py expt=mcmaster   # small parts: cotter-pin log 0
```

### Real-world trajectories

```python
python midastouch/filter/filter_real.py expt=ycb # default: 004_sugar_box log 0
python midastouch/filter/filter_real.py expt.obj_model=021_bleach_cleanser expt.log_id=2 # 021_bleach_cleanser log 2
```



## Codebook live demo

With your own [DIGIT](https://digit.ml/), you can simple plug in the sensor and experiment with the image to 3D and tactile codes visualizer. 


```python
python midastouch/filter/live_demo.py expt.obj_model=025_mug
```

<div align="center">
  <img src=".github/live_demo_mug.gif"
  width="60%">
</div>



## Folder structure
```bash
midastouch
├── bash          # bash scripts for filtering, codebook generation
├── config        # hydra config files 
├── contrib       # modified third-party code for TDN, TCN
├── eval          # select evaluation scripts 
├── filter        # filtering and live demo scripts
├── modules       # helper functions and classes
├── render        # DIGIT tactile rendering class
├── tactile_tree  # codebook scripts 
└── viz           # pyvista visualization 
```


## Bibtex

```
@inproceedings{suresh2022midastouch,
    title={{M}idas{T}ouch: {M}onte-{C}arlo inference over distributions across sliding touch},
    author={Suresh, Sudharshan and Si, Zilin and Anderson, Stuart and Kaess, Michael and Mukadam, Mustafa},
    booktitle = {Proc. Conf. on Robot Learning, CoRL},
    address = {Auckland, NZ},
    month = dec,
    year = {2022}
}
```


## License

The majority of MidasTouch is licensed under MIT license, however portions of the project are available under separate license terms: MinkLoc3D is licensed under the MIT license; FCRN-DepthPrediction is licensed under the BSD 2-clause license; pytorch3d is licensed under the BSD 3-clause license. Please see the [LICENSE](LICENSE) file for more information.



## Contributing

We actively welcome your pull requests! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for more info.