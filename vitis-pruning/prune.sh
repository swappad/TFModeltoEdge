WORKSPACE=./
MODEL=../trained_models/unet_model_next/
BASELINE_CKPT=${MODEL}/float_model.ckpt
INPUT_NODES="input_1"
OUTPUT_NODES="conv2d_13/truediv"

vai_p_tensorflow \
	 --action=prune \
	 --input_graph=${MODEL}/infer_graph.pb\
	 --input_nodes=${INPUT_NODES} \
	 --input_node_shapes="1,256,512,3" \
	 --input_ckpt=${BASELINE_CKPT} \
	 --output_graph=sparse_graph.pbtxt \
	 --output_ckpt=sparse.ckpt \
	 --workspace=/home/deephi/tf_models/research/slim \
	 --sparsity=0.1 \
	 --exclude="conv node names that excluded from pruning" \
	 --output_nodes="output node names of the network"
