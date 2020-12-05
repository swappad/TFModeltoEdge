WORKSPACE=./
MODEL=../trained_models/unet_model_next/
BASELINE_GRAPH=${MODEL}/infer_graph.pbtxt
BASELINE_CKPT=${MODEL}/float_model.ckpt
INPUT_NODES="input_1"
OUTPUT_NODES="conv2d_12/truediv"


vai_p_tensorflow \
	--action=ana \
	--input_graph=${BASELINE_GRAPH} \
	--input_ckpt=${BASELINE_CKPT} \
	--eval_fn_path=prune_eval.py \
	--target="accuracy" \
	--max_num_batches=10 \
	--workspace=${WORKSPACE} \
	--input_nodes=${INPUT_NODES} \
	--input_node_shapes="1,256,512,3" \
	--output_nodes=${OUTPUT_NODES} 

