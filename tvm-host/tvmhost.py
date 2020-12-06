import argparse
import os
from unet import unet

import logging
logging.basicConfig()
logger = logging.getLogger('pyxir')
logger.setLevel(logging.INFO)


import pyxir
import tvm

import tvm.relay as relay
import numpy as np
#from tvm.contrib.target import vitis_ai
from tvm.contrib import utils, graph_runtime
from tvm.relay.build_module import bind_params_by_name
#from tvm.relay.op.contrib.vitis_ai import annotation
from tvm.contrib.vai.relay_transform import PartitioningPass

from generator import data_generator
from tvm.contrib.vai import extern_accel
from tvm.contrib.vai.tvmruntime_util import TVMRuntimeUtil

import tensorflow.keras as keras
import tensorflow as tf


parser = argparse.ArgumentParser(description='import pretrained keras weights (.h5) to tvm relay')
parser.add_argument("--weights", help="path to h5 encoded weights")
args = parser.parse_args()

shape_dict = {'input_1': (1, 256, 512, 3)} # channels last

keras.backend.clear_session()
keras.backend.set_learning_phase(0)

model_keras = unet(input_shape=(256, 512, 3), num_classes=4, lr_init=1e-3, lr_decay=5e-4)
model_keras.load_weights(args.weights)


mod, params = relay.frontend.from_keras(model_keras, shape_dict, layout='NHWC')

tvm_target = 'llvm'
target = 'DPUCZDX8G-ultra96'

#mod["main"] = bind_params_by_name(mod["main"], params)
#mod = annotation(mod, params, target)
#mod = relay.transform.MergeCompilerRegions()(mod)
#mod = relay.transform.PartitionGraph()(mod)



#desired_layouts = {'nn.conv2d_transpose': ['NCHW', 'default']}
#seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
#with tvm.transform.PassContext(opt_level=3):
#    mod = seq(mod)
# 

TVM_OUTPUT_DIR = "./"
export_rt_mod_file = os.path.join(TVM_OUTPUT_DIR, "vitis_ai.rtmod")

#print("transform to NCHW")
#desired_layouts = {'nn.conv2d': ['NCHW', 'default']}
#
#seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
#                                relay.transform.ConvertLayout(desired_layouts)])
#with tvm.transform.PassContext(opt_level=3):
#    mod = seq(mod)

#print("build first time")

#with tvm.transform.PassContext(opt_level=3, config={'relay.ext.vitis_ai.options.target': target, 
#    'relay.ext.vitis_ai.options.export_runtime_module': export_rt_mod_file}):
#
#    lib = relay.build(mod, tvm_target, params=params)

# quantization
# module = graph_runtime.GraphModule(lib["default"](tvm.cpu()))

data = data_generator('./data.h5', 1, 'val')
def input_fn(iter):
    print("get data")
    tmp_data = next(data)[0]
    print(tmp_data.shape)
    return {'input_1': tmp_data}

print(input_fn(0))
print(input_fn(1))

os.environ['PX_QUANT_SIZE'] = "1"
#data = data_generator('./data.h5', 1, 'val')
#for i in range(1):
#    print("iteration", i)
#    module.set_input("input_1", next(data)[0])
#    module.run()


#print("quantization done")
#
#print("export tvm llvm lib")
#
#lib.export_library(os.path.join(TVM_OUTPUT_DIR, "tvm_lib.so"))
#

print("PartitioningPass")

# Export lib for aarch64 target
dpu_target = tvm.target.arm_cpu('ultra96')
lib_kwargs = {
     'fcompile': tvm.contrib.cc.create_shared,
     'cc': "/usr/aarch64-linux-gnu/bin/ld"
}
mod = PartitioningPass(target=target,params=params, inputs_func=input_fn, postprocessing=[])(mod)

print("build dpu relay")


with tvm.transform.PassContext(opt_level=3):
                               #config={'relay.ext.vitis_ai.options.load_runtime_module': export_rt_mod_file}):
     graph_json, lib_arm, params = relay.build(mod, dpu_target, params=params)

print("export dpu lib")
lib_arm.export_library('tvm_dpu_arm.so', **lib_kwargs)

with open(os.path.join(TVM_OUTPUT_DIR,"tvm_dpu_arm.json"),"w") as f:
    f.write(graph_json)

with open(os.path.join(TVM_OUTPUT_DIR,"tvm_dpu_arm.params"), "wb") as f:
    f.write(relay.save_param_dict(params))


