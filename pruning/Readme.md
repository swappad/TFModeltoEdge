# Pruning the model using Vitis AI Optmizer tools

(Note: Unfortunatly Vitis AI Optimizer tools are non-free products and require a license to run)

Official documentation can be found [here](https://www.xilinx.com/html_docs/vitis_ai/1_2/thf1576862844211.html).

This folder contains the scripts for pre-analysis of our model. ```vai_p_tensorflow``` requires a model function ```model_fn()``` and a data input function ```eval_input_fn()```. Both are implemented in ```prune_eval.py```. 

To collect data for evaluation we re-use the ```data_parser``` tools from [dhkim0225](https://github.com/dhkim0225/keras-image-segmentation/tree/eec48b96e9c7e8ac934268be56756eb8dac6ea6d), which we also used for training.
