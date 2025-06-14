import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, GlobalAveragePooling2D,
                                     GlobalMaxPooling2D, Add, Dense, Lambda)
from tensorflow.keras.models import Model

def build_unet(input_shape, num_classes, name_suffix=""):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottom of the U-Net
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Decoder
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(512, 2, activation='relu', padding='same')(up6)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same')(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same')(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same')(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    # Output layer
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(conv9)

    encoder_output = [conv1, conv2, conv3, conv4, conv5]

    encoder_model = Model(inputs=inputs, outputs=encoder_output, name=f"encoder{name_suffix}")
    decoder_model = Model(inputs=encoder_output, outputs=outputs, name=f"decoder{name_suffix}")

    return encoder_model, decoder_model


input_shape_model1 = (128, 128, 3)
num_classes_model1=1
encoder_model1, decoder_model1 = build_unet(input_shape_model1, num_classes_model1, "_image")

input_shape_model2 = (128, 128, 1)
num_classes_model2=1
encoder_model2, decoder_model2 = build_unet(input_shape_model2, num_classes_model2, "_mask")



def feats_sum(x):
  alphas=x[0]
  feats=x[1]
  alphas[0]
  alphas0=tf.expand_dims(alphas[0],axis=-1)
  alphas0=tf.expand_dims(alphas0,axis=-1)

  alphas1=tf.expand_dims(alphas[1],axis=-1)
  alphas1=tf.expand_dims(alphas1,axis=-1)

  alphas2=tf.expand_dims(alphas[2],axis=-1)
  alphas2=tf.expand_dims(alphas2,axis=-1)


  feat=alphas0*feats[0]+ alphas1*feats[1] + alphas2*feats[2]

  return feat

def features_attention(feat1,feat2,feat3,name="Image Attention Model"):
  concat = concatenate([feat1, feat2,feat3], axis=3,name=f"{name}_concat_img")
  avg_feat = GlobalAveragePooling2D(name=f"{name}_avgpool_img")(concat)
  max_feat = GlobalMaxPooling2D(name=f"{name}_maxpool_img")(concat)
  feat = Add(name=f"{name}_sum_img")([avg_feat, max_feat])
  feat = Dense(256,"relu", name=f"{name}_dense1_img")(feat)
  alphas= Dense(3,"softmax",name=f"{name}_dense2_img")(feat)
  alphas=tf.split(alphas,num_or_size_splits=3, axis=1,name="images_split")
  #alphas= tuple(alphas)
  feat= Lambda(feats_sum,name=f"{name}_featsum_img")([alphas,[feat1,feat2,feat3]])

  #feat=alphas[0]*feat1+ alphas[1]*feat2 + alphas[2]*feat3
  return feat

input_list1=Input((3,128,128,3),name="three_input_images")
image1,image2,image3 = tf.split(input_list1,num_or_size_splits=3,axis=1)
image1=tf.squeeze(image1,axis=1)
image2=tf.squeeze(image2,axis=1)
image3=tf.squeeze(image3,axis=1)

feats1=encoder_model1(image1)
feats2=encoder_model1(image2)
feats3=encoder_model1(image3)
feats=[]

for cnt,(feat1,feat2,feat3) in enumerate(zip(feats1,feats2,feats3)):

  feat= features_attention(feat1,feat2,feat3,name=f"conv{cnt}")
  feats.append(feat)
#attention_encoder_image_model= Model(inputs=[image1,image2,image3],outputs=feats)
attention_encoder_model= Model(inputs=[image1,image2,image3],outputs=feats,name="attention_image_encoder_model")
output1=decoder_model1(feats)


def feats_sum_masks(x):
    alphas = x[0]
    feats = x[1]
    alphas0 = tf.expand_dims(alphas[0], axis=-1)
    alphas0 = tf.expand_dims(alphas0, axis=-1)

    alphas1 = tf.expand_dims(alphas[1], axis=-1)
    alphas1 = tf.expand_dims(alphas1, axis=-1)

    output_feat = alphas0 * feats[0] + alphas1 * feats[1]

    return output_feat

# Features Attention
def features_attention_masks(feat_mask1, feat_mask2, name="Mask Attention Model"):
    concat = concatenate([feat_mask1, feat_mask2], axis=-1, name=f"{name}_concat_mask")
    avg_feat = GlobalAveragePooling2D(name=f"{name}_avgpool_mask")(concat)
    max_feat = GlobalMaxPooling2D(name=f"{name}_maxpool_mask")(concat)
    feat = Add(name=f"{name}_sum_mask")([avg_feat, max_feat])
    feat = Dense(256, "relu", name=f"{name}_dense1_mask")(feat)
    alphas = Dense(2, "softmax", name=f"{name}_dense2_mask")(feat)
    alphas_mask = tf.split(alphas, num_or_size_splits=2, axis=1, name="masks_split")
    output_feat = Lambda(feats_sum_masks, name=f"{name}_featsum_mask")([alphas_mask, [feat_mask1, feat_mask2]])

    return output_feat

input_list2 = Input((2, 128, 128, 1),name="two_input_masks")
mask1, mask2 = tf.split(input_list2, num_or_size_splits=2, axis=1)
mask1 = tf.squeeze(mask1, axis=1)
mask2 = tf.squeeze(mask2, axis=1)

feats_mask1 = encoder_model2(mask1)
feats_mask2 = encoder_model2(mask2)
feats_masks = []

for cnt, (feat_mask1, feat_mask2) in enumerate(zip(feats_mask1, feats_mask2)):
    output_feat = features_attention_masks(feat_mask1, feat_mask2, name=f"conv{cnt}")
    feats_masks.append(output_feat)

# Connect the input tensor to the outputs
attention_encoder_model_mask = Model(inputs=[mask1, mask2], outputs=feats_masks, name="attention_mask_encoder_model")
output2 = decoder_model2(feats_masks)

# Attention
def attentionBlock(feat_image: tf.Tensor, feat_mask: tf.Tensor, n_hidden: int = 256) -> tf.Tensor:
    # Global Average Pooling
    feat_image_1d = tf.keras.layers.GlobalAveragePooling2D()(feat_image)  # (batch_size, 64)
    feat_mask_1d = tf.keras.layers.GlobalAveragePooling2D()(feat_mask)    # (batch_size, 64)
    concat_final_feat = tf.concat([feat_image_1d, feat_mask_1d], axis=-1)  # (batch_size, 128)

    # Optional Dense Layer
    if n_hidden > 0:
        x_hidden = tf.keras.layers.Dense(n_hidden, activation="relu")(concat_final_feat)  # (batch_size, n_hidden)
    else:
        x_hidden = concat_final_feat

    # Get the attention weights for image and mask
    x_weight = tf.keras.layers.Dense(2, activation="softmax")(x_hidden)  # (batch_size, 2)
    x_weight = tf.expand_dims(tf.expand_dims(x_weight, 1), 1)  # (batch_size, 1, 1, 2)
    w0, w1 = tf.split(x_weight, num_or_size_splits=2, axis=-1)  # (batch_size, 1, 1, 1) each
    feat_final_sum = w0 * feat_image + w1 * feat_mask  # (batch_size, H, W, C)

    return feat_final_sum


# Model
attention_encoder_model= Model(inputs=[image1,image2,image3], outputs=feats, name="attention_encoder_image_model")
attention_encoder_model_mask = Model(inputs=[mask1, mask2], outputs=feats_masks, name="attention_encoder_mask_model")
attention_weighted_features = [attentionBlock(feat, feat_mask, n_hidden=0) for feat, feat_mask in zip(feats, feats_masks)]
output_combined=decoder_model1(attention_weighted_features)

combined_model = Model(inputs=[input_list1, input_list2], outputs=output_combined, name="combined_model")
combined_model.summary()
