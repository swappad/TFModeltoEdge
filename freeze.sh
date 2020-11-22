MODEL_PATH=./trained_models/unet_model_next_epoch17
freeze_graph \
	--input_graph 		${MODEL_PATH}/infer_graph.pb \
	--input_checkpoint 	${MODEL_PATH}/float_model.ckpt \
	--output_graph 		${MODEL_PATH}/frozen_graph.pb \
	--output_node_names conv2d_13/BiasAdd \
 	--input_binary 		true
	
