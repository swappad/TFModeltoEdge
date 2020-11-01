FROZEN_GRAPH=./trained_models/1210_unet_epoch_21/frozen_graph.pb
INPUT_NODES="input_1"
OUTPUT_NODES="conv2d_8/Conv2D"
OUTPUT_DIR="./quantize_result"

vai_q_tensorflow quantize \
	--input_frozen_graph ${FROZEN_GRAPH} \
	--input_nodes ${INPUT_NODES} \
	--input_shapes ?,256,512,3 \
	--output_nodes ${OUTPUT_NODES} \
	--input_fn graph_input_fn.calib_input \
	--weight_bit 8 \
	--activation_bit 8 \
	--method 1 \
	--calib_iter 10 \
	--simulate_dpu 1 \
	--output_dir ${OUTPUT_DIR}

