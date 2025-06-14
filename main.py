import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from UNT.data_preparation import (
    image_paths_list, mask_paths_list, prepare_dataset,
    model_path, metricsHistory_path, eval_path,
    num_epochs, num_earlyStop, lr, momentum
)
from UNT.MIA_UNet import combined_model, model_name

# IoU Function
def iou(y_true, y_pred):
    y_bin = tf.cast(tf.math.greater_equal(y_pred, 0.5), tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    shape = tf.shape(y_bin)
    batch = shape[0]
    y_bin = tf.reshape(y_bin, shape=[batch, -1])
    y_true = tf.reshape(y_true, shape=[batch, -1])
    intersect = tf.math.reduce_sum(y_bin * y_true, axis=-1)
    union = tf.math.reduce_sum(tf.cast(tf.math.greater(y_bin + y_true, 0), tf.float32), axis=-1)
    union = union + 1e-6
    iou_score = intersect / union
    return iou_score

if __name__ == "__main__":
    # Train Test Split
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths_list, mask_paths_list, test_size=0.2, random_state=42)
    train_dataset = prepare_dataset(train_images, train_masks)
    val_dataset = prepare_dataset(val_images, val_masks)

    # Binary Accuracy Early Stopping Checkpoint
    binary_accuracy = tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=num_earlyStop,  restore_best_weights=True)
    checkpoint_callback = ModelCheckpoint(filepath='combinedmodel_checkpoint.h5',save_best_only=True,save_weights_only=True,monitor='val_loss',mode='min',verbose=1)

    # Model Training with Hyper Parameters
    sgd_optimizer = SGD(learning_rate=lr, momentum=momentum)
    combined_model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=[iou, 'binary_accuracy'])
    history = combined_model.fit(train_dataset, validation_data=val_dataset,batch_size=5, epochs=num_epochs,callbacks=[early_stopping,checkpoint_callback])

    combined_model.save_weights(model_path)

    # Create Loss Accuracy History Figure
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    training_accuracy = history.history['binary_accuracy']
    validation_accuracy = history.history['val_binary_accuracy']
    training_iou = history.history['iou']
    validation_iou = history.history['val_iou']

    # Create subplots
    plt.figure(figsize=(12, 6))

    plt.subplot(131)
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(132)
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(133)
    plt.plot(training_iou, label='Training IOU')
    plt.plot(validation_iou, label='Validation IOU')
    plt.title('IOU')
    plt.xlabel('Epochs')
    plt.legend()

    plt.suptitle('Proposed Model Training and Validation Metrics', fontsize=16)
    # Save the figure
    plt.savefig(metricsHistory_path)
    plt.show()

    # Model Train Val Evaluation
    train_evaluation = combined_model.evaluate(train_dataset) #original

    print("Training Loss:", train_evaluation[0])
    print("Training IOU:", train_evaluation[1])
    print("Traininig Binary Accuracy:", train_evaluation[2])

    val_evaluation = combined_model.evaluate(val_dataset) #original

    print("Validation Loss:", val_evaluation[0])
    print("Validation IOU:", val_evaluation[1])
    print("Validation Binary Accuracy:", val_evaluation[2])

    # Save to text file
    with open(eval_path, 'w') as f:
        f.write(f"Model: {model_name}\n\n")
        f.write("Training Metrics:\n")
        f.write(f"Loss: {train_evaluation[0]}\n")
        f.write(f"IOU: {train_evaluation[1]}\n")
        f.write(f"Binary Accuracy: {train_evaluation[2]}\n\n")
        f.write("Validation Metrics:\n")
        f.write(f"Loss: {val_evaluation[0]}\n")
        f.write(f"IOU: {val_evaluation[1]}\n")
        f.write(f"Binary Accuracy: {val_evaluation[2]}\n")