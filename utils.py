import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from tensorflow.keras.callbacks import Callback


class MonitorTraining(Callback):

    def on_train_begin(self, _):
        self.loss = []
        self.accuracy = []
        self.val_loss = []
        self.val_accuracy = []
        plt.style.use("ggplot")

    def on_epoch_end(self, epoch, logs):
        self.loss.append(logs['loss'])
        self.accuracy.append(logs['accuracy'])
        self.val_loss.append(logs['val_loss'])
        self.val_accuracy.append(logs['val_accuracy'])

        if len(self.loss) > 1:
            clear_output(wait=True)
            self.figure, self.axes = plt.subplots(1, 2, figsize=(16, 9))
            current_epoch = len(self.loss)

            N = np.arange(1, current_epoch + 1)

            self.figure.suptitle(f"Epoch: {current_epoch}", fontsize=16)

            self.axes[0].plot(N, self.loss, color="red")
            self.axes[0].scatter(N, self.loss, color="red", label="Train")
            self.axes[0].plot(N, self.val_loss, color="blue")
            self.axes[0].scatter(N, self.val_loss, color="blue", label="Test")
            self.axes[0].set_title("Loss", fontsize=14)
            self.axes[0].set_xlabel("Epoch")
            self.axes[0].set_ylabel("Loss")
            self.axes[0].legend(fontsize=14)

            self.axes[1].plot(N, self.accuracy, color="red")
            self.axes[1].scatter(N, self.accuracy, color="red", label="Train")
            self.axes[1].plot(N, self.val_accuracy, color="blue")
            self.axes[1].scatter(N, self.val_accuracy, color="blue", label="Test")
            self.axes[1].set_title("Accuracy", fontsize=14)
            self.axes[1].set_xlabel("Epoch")
            self.axes[1].set_ylabel("Accuracy")
            self.axes[1].legend(fontsize=14)

            plt.show()


def create_model_checkpoint(model_name, save_path="models"):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                              monitor="val_accuracy",
                                              verbose=0,
                                              save_best_only=True)


def reduce_lr(factor=0.1, patience=3):
    return tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",
                                                factor=factor,
                                                patience=patience,
                                                verbose=1)


def predict_class(model, test_images, test_labels, class_names):

    rand_int = np.random.randint(len(test_images))

    pred = model.predict(tf.expand_dims(test_images[rand_int], axis=0), verbose=0)
    pred_class = pred.argmax()

    fig, axs = plt.subplots(1, 2, figsize=(13, 7))

    axs[0].imshow(test_images[rand_int])
    axs[0].axis("off")
    axs[0].set_title(f"True label: {class_names[test_labels[rand_int]]},    Predicted label: {class_names[pred_class]}")

    axs[1].bar(list(class_names.values()), pred[0])
    axs[1].grid(True)
    axs[1].set_ylim([-0.03, 1.03])
    axs[1].set_xticklabels(list(class_names.values()), rotation=45)


def get_images_and_labels_from_dataset(ds):
    images = np.asarray([image for image, label in ds])
    labels = np.asarray([label for image, label in ds])
    return images, labels
