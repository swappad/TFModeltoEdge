MODEL_PATH=./trained_models/unet_model_next
FROZEN_GRAPH=${MODEL_PATH}/frozen_graph.pb
OPTIMIZED_GRAPH=${MODEL_PATH}/optimized_graph.pb
INFER_GRAPH=${MODEL_PATH}/infer_graph.pb
INPUT_NODES="input_1"
OUTPUT_NODES="conv2d_13/BiasAdd"
OUTPUT_DIR="./quantize_result"


vai_q_tensorflow quantize \
	--input_frozen_graph ${FROZEN_GRAPH} \
	--input_nodes ${INPUT_NODES} \
	--input_shapes ?,256,512,3 \
	--output_nodes ${OUTPUT_NODES} \
	--input_fn graph_input_fn.calib_input \
	--method 1 \
	--weight_bit 8 \
	--activation_bit 8 \
	--calib_iter 20\
	--simulate_dpu 1 \
	--output_dir ${OUTPUT_DIR} \
	--dump_float 1

