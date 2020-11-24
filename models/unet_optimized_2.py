from keras.models import Model
from keras.layers import Input, Concatenate, MaxPooling2D, UpSampling2D, Activation, SeparableConv2D, Conv2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K


def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def unet(num_classes, input_shape, lr_init, lr_decay, vgg_weight_path=None):
    img_input = Input(input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    # x = SeparableConv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    # x = SeparableConv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    block_1_out = Activation('relu')(x)

    x = MaxPooling2D()(block_1_out)

    # Block 2
    # x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    block_2_out = Activation('relu')(x)

    x = MaxPooling2D()(block_2_out)

    # Block 3
    # x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = SeparableConv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = SeparableConv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = SeparableConv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    block_3_out = Activation('relu')(x)

    x = MaxPooling2D()(block_3_out)

    # Block 4
    # x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
    x = SeparableConv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = SeparableConv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = SeparableConv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    block_4_out = Activation('relu')(x)

    x = MaxPooling2D()(block_4_out)

    # Block 5
    # x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
    x = SeparableConv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    x = SeparableConv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
    x = SeparableConv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for_pretrained_weight = MaxPooling2D()(x)

    # Load pretrained weights.
    if vgg_weight_path is not None:
        vgg16 = Model(img_input, for_pretrained_weight)
        vgg16.load_weights(vgg_weight_path, by_name=True)

    # UP 1
    x = UpSampling2D(size=(2, 2),data_format="channels_last")(x)
    #x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
    # x  = Conv2D(512, kernel_size=2, data_format="channels_last", activation="relu", padding="same")(x)
    x  = SeparableConv2D(512, kernel_size=2, data_format="channels_last", activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)

    x = Concatenate(axis=3)([x, block_4_out])
    # x = Conv2D(512, (3, 3), padding='same')(x)
    x = SeparableConv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Conv2D(512, (3, 3), padding='same')(x)
    x = SeparableConv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 2
    x = UpSampling2D(size=(2, 2),data_format="channels_last")(x)
    #x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    # x  = Conv2D(256, kernel_size=2, data_format="channels_last", activation="relu", padding="same")(x)
    x  = SeparableConv2D(256, kernel_size=2, data_format="channels_last", activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)

    x = Concatenate(axis=3)([x, block_3_out])
    # x = Conv2D(256, (3, 3), padding='same')(x)
    x = SeparableConv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Conv2D(256, (3, 3), padding='same')(x)
    x = SeparableConv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 3
    x = UpSampling2D(size=(2, 2),data_format="channels_last")(x)
    #x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    # x  = Conv2D(128, kernel_size=2, data_format="channels_last", activation="relu", padding="same")(x)
    x  = SeparableConv2D(128, kernel_size=2, data_format="channels_last", activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)

    x = Concatenate(axis=3)([x, block_2_out])
    # x = Conv2D(128, (3, 3), padding='same')(x)
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Conv2D(128, (3, 3), padding='same')(x)
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 4
    x = UpSampling2D(size=(2, 2),data_format="channels_last")(x)
    #x = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same', data_format="channels_last")(x)
    # x  = Conv2D(64, kernel_size=2, data_format="channels_last", activation="relu", padding="same")(x)
    x  = SeparableConv2D(64, kernel_size=2, data_format="channels_last", activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)

    x = Concatenate(axis=3)([x, block_1_out])
    # x = Conv2D(64, (3, 3), padding='same')(x)
    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Conv2D(64, (3, 3), padding='same')(x)
    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # last conv
    # x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)
    x = SeparableConv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)

    model = Model(img_input, x)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    return model
