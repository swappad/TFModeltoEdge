MODEL_PATH=./trained_models/unet_model_next
OUTPUT_NAME=conv2d_13/BiasAdd

freeze_graph \
	--input_graph 		${MODEL_PATH}/infer_graph.pb \
	--input_checkpoint 	${MODEL_PATH}/float_model.ckpt \
	--output_graph 		${MODEL_PATH}/frozen_graph.pb \
	--output_node_names ${OUTPUT_NAME} \
 	--input_binary 		true
	
