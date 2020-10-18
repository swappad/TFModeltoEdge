# TF Keras to Xilinx Zynq (Ultra96v2) design flow
This repo contain all files for compilation and deployment.

## List of files:
-----------
|File name 	| description | comment |
----------- | ---------- | -------- |
|dpu.hwh | hardware description file for Ultra96 dev Board| - |
|docker_run.sh | starts the docker for VITIS AI with required parameters | ! `conda activate vitis-ai-tensorflow` |
|h5topb.py | loads weights (hdf5 format) and freezes the model graph | ! run inside VITIS AI docker to meet the required TF version 1.15 |
|quantize.sh | script to quantize the frozen model| run inside docker |
|unet.py | Unet model from [dhkim0225](https://github.com/dhkim0225/keras-image-segmentation.git) |  insert `model.trainable=False` before `model.compile`


