import random
import numpy as np
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Concatenate, Conv2DTranspose, BatchNormalization
from keras.layers import BatchNormalization, Activation, Dropout

def build_embedding(param, inp, s):
    #network = eval(param["network_name"])
    #base = network(weights = 'imagenet', include_top = False)
    #for layer in base.layers:
    #    layer.name = layer.name + str('_')+ str(s)
    
    #for layer in base.layers:
        #print(layer.name)

    #conv_1 = Conv2D(3, kernel_size=3,strides = 2, activation='relu', name = s+'_conv1')(inp)
    #conv_2 = Conv2D(3, kernel_size=3, strides = 3, activation='relu', name = s+'_conv2')(conv_1)
    #embedding = Conv2D(3, kernel_size=3, strides = 3, activation='relu', name = s+'_emb')(conv_2)
    base = ResNet50(weights = 'imagenet', input_shape = (param["inp_dims"]), include_top = False)
    if s == 'embedding':
        embedding = Model(inputs=base.input, outputs=base.get_layer('activation_49').output, name='resnet50_'+s)(inp)
    if s == 'sembedding':
        embedding = Model(inputs=base.input, outputs=base.get_layer('activation_98').output, name='resnet50_'+s)(inp)
    if s == 'tembedding':
        embedding = Model(inputs=base.input, outputs=base.get_layer('activation_147').output, name='resnet50_'+s)(inp)

    #embedding = base(inp)
    return embedding

def build_classifier(param, embedding):
    flat = Flatten(name = 'class_flat')(embedding)
    dense1 = Dense(2048, name = 'class_dense1')(flat)
    bn1 = BatchNormalization(name = 'class_bn1')(dense1)
    act1 = Activation('relu', name = 'class_act1')(bn1)
    drop2 = Dropout(param["drop_classifier"], name = 'class_drop1')(act1)

    dense2 = Dense(2048, name = 'class_dense2')(drop2)
    bn2 = BatchNormalization(name = 'class_bn2')(dense2)
    act2 = Activation('relu', name = 'class_act2')(bn2)
    drop2 = Dropout(param["drop_classifier"], name = 'class_drop2')(act2)

    densel = Dense(param["source_label"].shape[1], name = 'class_dense_last')(drop2)
    bnl = BatchNormalization(name = 'class_bn_last')(densel)
    actl = Activation('softmax', name = 'class_act_last')(bnl)
    return actl

def build_discriminator(param, embedding):
    flat = Flatten(name = 'dis_flat')(embedding)
    dense1 = Dense(2048, name = 'dis_dense1')(flat)
    bn1 = BatchNormalization(name='dis_bn1')(dense1)
    act1 = Activation('relu', name = 'dis_act1')(bn1)
    drop1 = Dropout(param["drop_discriminator"], name = 'dis_drop1')(act1)

    dense2 = Dense(2048, name = 'dis_dense2')(drop1)
    bn2 = BatchNormalization(name='dis_bn2')(dense2)
    act2 = Activation('relu', name = 'dis_act2')(bn2)
    drop2 = Dropout(param["drop_discriminator"], name = 'dis_drop2')(act2)

    densel = Dense(1, name = 'dis_dense_last')(drop2)
    bnl = BatchNormalization(name = 'dis_bn_last')(densel)
    actl = Activation('sigmoid', name = 'dis_act_last')(bnl)
    return actl

def build_decoder(param, s_embedding, p_embedding, s):
    con = Concatenate(axis=3, name = s + '_dcon')([s_embedding, p_embedding])

    convt1 = Conv2DTranspose(256, (3, 3), strides=2, padding='same', name = s + '_dconvt1')(con)
    bn1 = BatchNormalization(name = s + '_dbn1')(convt1)
    act1 = Activation('relu', name =s+ '_act1')(bn1)

    convt2 = Conv2DTranspose(128,(3, 3), strides=2, padding='same', name = s + '_dconvt2')(act1)
    bn2 = BatchNormalization(name = s + '_dbn2')(convt2)
    act2 = Activation('relu', name =s+ '_act2')(bn2)

    convt3 = Conv2DTranspose(128,(3, 3), strides=2, padding='same', name = s + '_dconvt3')(act2)
    bn3 = BatchNormalization(name = s + '_dbn3')(convt3)
    act3 = Activation('relu', name =s+ '_act3')(bn3)

    convt4 = Conv2DTranspose(64,(3, 3), strides=2, padding='same', name = s + '_dconvt4')(act3)
    bn4 = BatchNormalization(name = s + '_dbn4')(convt4)
    act4 = Activation('relu', name =s+ '_act4')(bn4)

    convt5 = Conv2DTranspose(3,(3, 3), strides=2, padding='same', name = s + '_dconvt5')(act4)
    bn5 = BatchNormalization(name = s + '_dbn5')(convt5)
    act5 = Activation('sigmoid', name =s+ '_act5')(bn5)
    return act5 

def build_enc_dec(inp, inp2, decoding, s):
    comb_model = Model(inputs = [inp, inp2], outputs = [decoding], name = s )
    return comb_model

def build_combined_classifier(inp, classifier):
    comb_model = Model(inputs = inp, outputs = [classifier])
    return comb_model

def build_combined_discriminator(inp, discriminator):
    comb_model = Model(inputs = inp, outputs = [discriminator])
    return comb_model

def build_combined_model(inp, comb):
    comb_model = Model(inputs = inp, outputs = comb)
    return comb_model
