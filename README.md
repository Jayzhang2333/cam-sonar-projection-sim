# Sonar–Camera Simulator

A simple sonar camera simulator. It takes in an RGB and Depth image and renders sonar image and test different way of projecting sonar data into camera'a frame. User can cahneg camera and sonar intrinsics as well as extrinsics between camera and sonar.

## Install
```bash
conda env create -f environment.yml
conda activate sonarcam-sim-qt5
pip install -e .
python -m sonarcam.app
```
