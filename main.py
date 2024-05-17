import cv2

from AttentionUNet import AttentionUNet
from InpaintingTrainer import InpaintingTrainer
from Preprocess import preprocess_images
from UNetLikeModel import UNetLikeModel
from PIL import Image
import os
from tensorflow.keras import layers, models, optimizers
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.utils import shuffle

# Directories containing train and test data
train_input_dir = "../image-processing-api/inpainting/train_input"
train_mask_dir = "../image-processing-api/inpainting/train_mask"
train_gt_dir = "../image-processing-api/inpainting/train_output"

test_input_dir = "../image-processing-api/inpainting/test_input"
test_mask_dir = "../image-processing-api/inpainting/test_mask"
test_gt_dir = "../image-processing-api/inpainting/test_output"

# Get paths for train and test data
train_input_paths = [os.path.join(train_input_dir, filename) for filename in
                     sorted(os.listdir(train_input_dir))[:10000]]
train_mask_paths = [os.path.join(train_mask_dir, filename) for filename in sorted(os.listdir(train_mask_dir))[:100]]
train_gt_paths = [os.path.join(train_gt_dir, filename) for filename in sorted(os.listdir(train_gt_dir))[:100]]

test_input_paths = [os.path.join(test_input_dir, filename) for filename in sorted(os.listdir(test_input_dir))[:100]]
test_mask_paths = [os.path.join(test_mask_dir, filename) for filename in sorted(os.listdir(test_mask_dir))[:100]]
test_gt_paths = [os.path.join(test_gt_dir, filename) for filename in sorted(os.listdir(test_gt_dir))[:100]]

# We separate validation from test set because the dataset loading takes too much time. We used first 1000 for testing and last 1000 for validation.
val_input_paths = [os.path.join(test_input_dir, filename) for filename in sorted(os.listdir(test_input_dir))[-100:]]
val_mask_paths = [os.path.join(test_mask_dir, filename) for filename in sorted(os.listdir(test_mask_dir))[-100:]]
val_gt_paths = [os.path.join(test_gt_dir, filename) for filename in sorted(os.listdir(test_gt_dir))[-100:]]

# Load and preprocess train data
train_input_images, train_masks, train_gts = preprocess_images(train_input_paths, train_mask_paths, train_gt_paths)
# Load and preprocess train data
val_input_images, val_masks, val_gts = preprocess_images(val_input_paths, val_mask_paths, val_gt_paths)
# Load and preprocess test data
test_input_images, test_masks, test_gts = preprocess_images(test_input_paths, test_mask_paths, test_gt_paths)

# Combine input images, masks, and ground truth images
train_input_data = np.concatenate((train_input_images, train_masks), axis=-1)
test_input_data = np.concatenate((test_input_images, test_masks), axis=-1)
val_input_data = np.concatenate((val_input_images, val_masks), axis=-1)

# You can choose the model you want to use by changing the model_choice variable
model_choice = ""
model = None

if model_choice == "CustomModel":
    model = UNetLikeModel().model
    model.compile(optimizer=optimizers.Adam(), loss=MeanSquaredError())
    model.fit(train_input_data, train_gts, epochs=50, batch_size=32, validation_data=(val_input_data, val_gts))
elif model_choice == "AttentionUNetMSE":
    model = AttentionUNet()
    model.compile(optimizer=optimizers.Adam(), loss=MeanSquaredError())
    model.fit(train_input_data, train_gts, epochs=50, batch_size=32, validation_data=(val_input_data, val_gts))
elif model_choice == "AttentionUNetPerceptual":
    model = AttentionUNet()


    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.Adam()

    # Define callbacks
    test_index_to_print = None  # Index of the test image to print, if you like you can choose the index of the image to print per epoch results

    # Create InpaintingTrainer instance
    trainer = InpaintingTrainer(model, train_input_data, train_gts, val_input_data, val_gts, optimizer, epochs=50,
                                batch_size=32,
                                test_input_data=test_input_data, test_index_to_print=test_index_to_print)

    # Train the model
    model = trainer.train()

preds = model.predict(test_input_data)

# Save the predictions if you like
"""for i, pred in enumerate(preds):
    pred_image = Image.fromarray((pred * 255).astype(np.uint8))
    if not os.path.exists("predictions"):
        os.makedirs("predictions")
    pred_image.save(f"predictions/pred_{i}.png")"""