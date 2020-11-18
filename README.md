# TF Keras to Xilinx Zynq (Ultra96v2) design flow
This repo contains all files for compilation and deployment.

## List of files:
-----------
|File name 	| description | comment |
----------- | ---------- | -------- |
|dpu.hwh | hardware description file for Ultra96 dev Board| - |
|docker_run.sh | starts the docker for VITIS AI with required parameters | ! `conda activate vitis-ai-tensorflow` |
|h5topb.py | loads weights (hdf5 format) and freezes the model graph | ! run inside VITIS AI docker to meet the required TF version 1.15 |
|quantize.sh | script to quantize the frozen model| run inside docker |
|unet.py | Unet model from [dhkim0225](https://github.com/dhkim0225/keras-image-segmentation.git) |  insert `model.trainable=False` before `model.compile`
|Ultra96.json | from DPU-PYNQ repo |- |
|Ultra96.dcf | generated from Ultra96 using `dlet` | inside docker |
|compile.sh | script for compilation | inside docker |
|dump.sh| dump constants for each layer after quantization | use unknown |

## Step 1: export and freeze TF Model
`h5topb.py` -> adjust target path and filename and file path to hdf5 encoded weights 

! run inside VITIS AI docker or with tensorflow v1.15

## Step 2: Quantization
`quantize.sh` -> provide a subset of the image dataset previously used for training or validation and adjust path in `graph_input_fn.py` accordingly. This python script is required for calibration of the quantized model.

! Make sure that output and input labels are correct for `quantize.sh`and `graph_input_fn.py`

## Step 3: Compilation
`compile.sh` -> Ultra96.json can be retrieved through DPU-PYNQ compilation, but must point to an existing Ultra96.dcf file


## Troubleshooting
### Freezing fails:
* `IndexError: list index out of range` 

	Make sure the docker image is modified such that conda has correct `keras==2.2.4` package installed:
	```
	./docker_run.sh xilinx/vitis-ai-cpu:latest
	sudo su
	conda activate vitis-ai-tensorflow
	conda install keras==2.2.4  # compatible with TF 1.15
	conda deactivate
	exit
	conda activate vitis-ai-tensorflow
	```
	Stage your modifications for the next docker restart:
	```
	sudo docker commit -m "<message>" <image_id> xilinx/vitis-ai-cpu:latest
	```





