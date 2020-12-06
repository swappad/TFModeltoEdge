MODEL_PATH=./trained_models/unet_model_optimized_2
FROZEN_GRAPH=${MODEL_PATH}/frozen_graph.pb
INPUT_NODES="input_1"
OUTPUT_NODES="separable_conv2d_24/BiasAdd"
OUTPUT_DIR="./quantize_result"


vai_q_tensorflow quantize \
	--input_frozen_graph ${FROZEN_GRAPH} \
	--input_nodes ${INPUT_NODES} \
	--input_shapes ?,256,512,3 \
	--output_nodes ${OUTPUT_NODES} \
	--input_fn graph_input_fn.input_fn\
	--method 1 \
	--weight_bit 8 \
	--activation_bit 8 \
	--calib_iter 2\
	--simulate_dpu 1 \
	--output_dir ${OUTPUT_DIR} \
	--dump_float 1

