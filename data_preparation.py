import tensorflow as tf
import os
main_dir = './'

image_folder = 'batch2_croppedimages'
mask_folder = 'batch2_tensorflow_originalsize_croppedmasks'

model_name = 'combined_unet_transformer_v1.h5'
metricsHistory_name = 'combined_unet_transformer_v1.png'
eval_name = 'combined_unet_transformer_v1.txt'

batch_size = 5
lr = 0.001
momentum = 0.9

num_epochs = 2
num_earlyStop = 5

# Combine file paths
images_path = os.path.join(main_dir, image_folder)
masks_path = os.path.join(main_dir, mask_folder)
model_path = os.path.join(main_dir, model_name)
metricsHistory_path = os.path.join(main_dir, metricsHistory_name)
eval_path = os.path.join(main_dir, eval_name)


# Create Sliding Window Txt File
def create_sliding_window_grouped_filenames_by_id(image_folder, mask_folder, output_txt_path, window_size):

    def extract_plant_id(filename):
        return filename.split('_')[-1].split('.')[0]

    image_filenames = sorted(os.listdir(image_folder))
    mask_filenames = sorted(os.listdir(mask_folder))

    images_by_id = {}
    masks_by_id = {}

    for fname in image_filenames:
        plant_id = extract_plant_id(fname)
        images_by_id.setdefault(plant_id, []).append(fname)

    for fname in mask_filenames:
        plant_id = extract_plant_id(fname)
        masks_by_id.setdefault(plant_id, []).append(fname)

    with open(output_txt_path, 'w') as txt_file:
        for plant_id, image_group in images_by_id.items():
            mask_group = masks_by_id.get(plant_id, [])

            for i in range(len(image_group) - window_size + 1):
                combined_group = image_group[i:i+window_size] + mask_group[i:i+window_size]
                txt_file.write(' '.join(combined_group) + '\n')

    print(f"Grouped filenames by plant ID saved to {output_txt_path}")

create_sliding_window_grouped_filenames_by_id(images_path, masks_path, './combined_filenames_samePlantID.txt', 3)

with open("./combined_filenames_samePlantID.txt", "r") as f:
    lines = f.read().splitlines()
    grouped_filenames = [line.split(' ') for line in lines]  # by space

image_paths_list = [[os.path.join(image_folder, fname) for fname in group[:3]] for group in grouped_filenames]
mask_paths_list = [[os.path.join(mask_folder, fname) for fname in group[3:]] for group in grouped_filenames]

def load_and_preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = img / 255.0
    return img

def load_and_preprocess_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, [128, 128])
    mask = tf.cast(mask > 127.5, tf.float32) #*****#
    return mask

def process_paths(img1_path, img2_path, img3_path, mask1_path, mask2_path, target_mask_path):
    img1 = load_and_preprocess_image(img1_path)
    img2 = load_and_preprocess_image(img2_path)
    img3 = load_and_preprocess_image(img3_path)
    mask1 = load_and_preprocess_mask(mask1_path)
    mask2 = load_and_preprocess_mask(mask2_path)
    target_mask = load_and_preprocess_mask(target_mask_path)
    return (tf.stack([img1, img2, img3], axis=0), tf.stack([mask1, mask2], axis=0)), target_mask

def prepare_dataset(image_paths_list, mask_paths_list):
    image_paths_dataset = tf.data.Dataset.from_tensor_slices(image_paths_list)
    mask_paths_dataset = tf.data.Dataset.from_tensor_slices(mask_paths_list)

    dataset = tf.data.Dataset.zip((image_paths_dataset, mask_paths_dataset))
    dataset = dataset.map(lambda x, y: process_paths(x[0], x[1], x[2], y[0], y[1], y[2]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=len(image_paths_list))
    dataset = dataset.batch(5)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset