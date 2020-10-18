OUTPUT_DIR='./compile_result'
KERNEL_NAME='Unet'

vai_c_tensorflow \
	--frozen_pb ./quantize_result/deploy_model.pb \
	--arch ./Ultra96.json \
	--output_dir ${OUTPUT_DIR} \
	--net_name ${KERNEL_NAME} \
	--options "{'mode':'debug','dump':'fused_graph_info'}"

