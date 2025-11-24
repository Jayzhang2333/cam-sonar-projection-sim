# Sonar–Camera Simulator

A simple sonar–camera simulator. Users can change the camera and sonar intrinsics, as well as the extrinsics between the camera and sonar. The visualizer shows how the sensors are oriented and how their fields of view overlap. The second figure shows how a sonar pixel is projected into the camera’s view.

![alt text](<asset/demo.png>)

## Install
```bash
conda env create -f environment.yml
conda activate sonarcam-sim-qt5
pip install -e .
python -m sonarcam.app
```
