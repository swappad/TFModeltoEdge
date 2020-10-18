vai_q_tensorflow dump \
	--input_frozen_graph ./quantize_result/quantize_eval_model.pb \
	--input_fn graph_input_fn.calib_input \
	--max_dump_batches 1 \
	--dump_float 0 \
	--output_dir quantize_result \

