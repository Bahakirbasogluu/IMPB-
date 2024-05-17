from tensorflow.keras import layers, models, optimizers
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class InpaintingTrainer:
    def __init__(self, model, train_data, train_gts, val_data, val_gts, optimizer, epochs, batch_size, test_input_data, test_index_to_print):
        self.model = model
        self.train_data = train_data
        self.train_gts = train_gts  # Added train_gts
        self.val_data = val_data
        self.val_gts = val_gts  # Added val_gts
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_input_data = test_input_data
        self.test_index_to_print = test_index_to_print

    def train(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            self._train_epoch()
            self._validate_epoch()
            self._log_predictions(epoch)
            print()
        return self.model

    def _train_epoch(self):
        num_batches = len(self.train_data) // self.batch_size
        total_loss = 0.0

        for batch in range(num_batches):
            batch_start = batch * self.batch_size
            batch_end = (batch + 1) * self.batch_size
            inputs = self.train_data[batch_start:batch_end]
            targets = self.train_gts[batch_start:batch_end]

            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)

                # Load pre-trained VGG19 model (or any other suitable pre-trained model)
                vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
                vgg.trainable = False

                # Define layers from which to extract features for perceptual loss
                content_layers = ['block4_conv4']

                # Define a model that outputs the features from the selected layers
                content_model = tf.keras.Model(inputs=vgg.input, outputs=[vgg.get_layer(layer).output for layer in content_layers])

                # Define perceptual loss function
                # Scale images to the range expected by VGG model
                y_true = targets * 255.0
                y_pred = predictions * 255.0

                # Get features from VGG for both true and predicted images
                true_features = content_model(y_true)
                pred_features = content_model(y_pred)

                # Compute MSE loss between features
                loss = 0.0
                for true_feature, pred_feature in zip(true_features, pred_features):
                    loss += tf.reduce_mean(tf.square(true_feature/255.0 - pred_feature/255.0))

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            total_loss += loss.numpy()

        avg_loss = total_loss / num_batches
        print(f"Train Loss: {avg_loss:.4f}")

    def _validate_epoch(self):
        num_batches = len(self.val_data) // self.batch_size
        total_loss = 0.0

        for batch in range(num_batches):
            batch_start = batch * self.batch_size
            batch_end = (batch + 1) * self.batch_size
            inputs = self.val_data[batch_start:batch_end]
            targets = self.val_gts[batch_start:batch_end]

            predictions = self.model(inputs, training=False)

            # Load pre-trained VGG19 model (or any other suitable pre-trained model)
            vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
            vgg.trainable = False

            # Define layers from which to extract features for perceptual loss
            content_layers = ['block4_conv4']

                # Define a model that outputs the features from the selected layers
            content_model = tf.keras.Model(inputs=vgg.input, outputs=[vgg.get_layer(layer).output for layer in content_layers])

                # Define perceptual loss function
                # Scale images to the range expected by VGG model
            y_true = targets * 255.0
            y_pred = predictions * 255.0

                # Get features from VGG for both true and predicted images
            true_features = content_model(y_true)
            pred_features = content_model(y_pred)

                # Compute MSE loss between features
            loss = 0.0
            for true_feature, pred_feature in zip(true_features, pred_features):
                loss += tf.reduce_mean(tf.square(true_feature/255.0 - pred_feature/255.0))

                #loss = self.loss_fn(targets, predictions, inputs[:, :, :, 3])

            total_loss += loss.numpy()

        avg_loss = total_loss / num_batches
        print(f"Validation Loss: {avg_loss:.4f}")

    def _log_predictions(self, epoch):
        if self.test_index_to_print is not None:
            test_input = np.expand_dims(self.test_input_data[self.test_index_to_print], axis=0)
            inpainted_image = self.model.predict(test_input)
            inpainted_image = inpainted_image.reshape(inpainted_image.shape[1:])
            print(f'Epoch {epoch+1} - Test Image {self.test_index_to_print + 1} Prediction:')
            plt.imshow(inpainted_image)
            plt.axis('off')
            plt.show()