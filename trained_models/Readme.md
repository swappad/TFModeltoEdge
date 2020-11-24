## Log for trained models :
| date 	| file name 			| loss 	 | description | comment |
--------|-----------------------|--------|-------------|---------|
| 07/10 | 0710_unet_epoch_3.h5 	| 0.1504 | default Unet von [dhkim0225](https://github.com/dhkim0225/keras-image-segmentation.git) | naja |
| 12/10 | 0710_unet_epoch_21.h5 | 0.0665 | " | better |
| 17/11 | unet_model_next.h5    | --     | dhkim0225 Unet, Conv2dTranspose replaced by Upscaling/Convolution | trained with pretrained vgg16 weights |
| 23/11 | unet_model_optimized_2.h5 | -- | previous network, but replaced all but the first two conv2d layers by separable conv2d layers | accuracy not the same as before yet | Epoch 9 |
