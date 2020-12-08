# TF Keras to Xilinx Zynq (Ultra96v2) design flow
This repo contains all files for compilation and deployment.

## List of files:
-----------
|File name 	| description | comment |
----------- | ---------- | -------- |
|hw_files | hardware description files for DPU on Ultra96 v2| - |
|docker_run.sh | starts the docker for VITIS AI with required parameters | ! `conda activate vitis-ai-tensorflow` |
|h5topb.py | loads weights (hdf5 format) and converts to TensorFlow .pb format| run inside VITIS AI docker for TF version 1.15 |
| freeze.sh | script to freeze the converted model | run inside docker |
|quantize.sh | script to quantize the frozen model| run inside docker |
|unet.py | Unet model from [dhkim0225](https://github.com/dhkim0225/keras-image-segmentation.git) | - |
|compile.sh | script for compilation | inside docker |

## Step 1: Convert Model to .pb format
`h5topb.py` -> provide file path to hdf5 encoded weights 

! run inside VITIS AI docker or with tensorflow v1.15 and ```keras==2.2.4```

## Step 2: Freeze the converted Model
```freeze.sh``` -> changed input and output node names accordingly

## Step 3: Quantization
`quantize.sh` -> provide a subset of the image dataset previously used for training or validation and adjust path in `graph_input_fn.py` accordingly. This python script is required for calibration of the quantized model.

! Make sure that output and input labels are correct for `quantize.sh`and `graph_input_fn.py`

## Step 4: Compilation
`compile.sh` -> DPU hardware files can be retrieved through DPU-PYNQ build. 


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





