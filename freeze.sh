freeze_graph \
	--input_graph 		./trained_models/1210_unet_epoch_21/infer_graph.pb \
	--input_checkpoint 	./trained_models/1210_unet_epoch_21/float_model.ckpt \
	--output_graph 		./trained_models/1210_unet_epoch_21/frozen_graph.pb \
	--output_node_names conv2d_9/BiasAdd \
 	--input_binary 		true
	
