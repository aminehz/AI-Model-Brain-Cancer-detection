

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Define constants for the data
batch_size = 32         # Number of samples per gradient update
img_height = 180        # Height of the input images
img_width = 180         # Width of the input images

data_train_dir = "data"

# Load the training dataset from the directory with a 80-20 train-validation split
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_train_dir,
    validation_split=0.2,          # Reserve 20% of data for validation
    subset="training",             # Subset used for training
    seed=123,                      # Seed for randomization
    image_size=(img_height, img_width),  # Resize images to (180, 180)
    batch_size=batch_size          # Number of images to return in each batch
)

# Load the validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_train_dir,
    validation_split=0.2,
    subset="validation",           # Subset used for validation
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)



# Get the class names from the dataset
class_names = train_ds.class_names
print(class_names)  # Output the class names

# Display a few sample images from the training dataset
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(2):  # Take two batches of images
    for i in range(9):  # Display 9 images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))  # Convert to uint8 for display
        plt.title(class_names[labels[i]])  # Display the corresponding class name
        plt.axis("off")  # Turn off axis

# Check the shape of the image and label batches
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)  # Output shape: (batch_size, img_height, img_width, channels)
    print(labels_batch.shape)  # Output shape: (batch_size,)
    break

# Prefetch and cache data for better performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Create a normalization layer to scale pixel values to [0, 1]
normalization_layer = layers.Rescaling(1./255)

# Apply the normalization layer to the training dataset
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Extract and check the min and max values of the first image
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print("Minimum value:", np.min(first_image), "Maximum value:", np.max(first_image))

# Define the number of classes in the dataset
num_classes = len(class_names)

# Build a simple CNN model
model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),  # Rescale input images
    layers.Conv2D(16, 3, padding='same', activation='relu'),          # 16 filters, 3x3 kernel
    layers.MaxPooling2D(),                                            # Downsample the input
    layers.Conv2D(32, 3, padding='same', activation='relu'),          # 32 filters, 3x3 kernel
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),          # 64 filters, 3x3 kernel
    layers.MaxPooling2D(),
    layers.Flatten(),                                                 # Flatten the input
    layers.Dense(128, activation='relu'),                             # Fully connected layer
    layers.Dense(num_classes)                                         # Output layer
])

# Compile the model with Adam optimizer and Sparse Categorical Crossentropy loss
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)

model.summary()  # Print the model architecture

# Train the model for 10 epochs
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Extract and plot training and validation accuracy/loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the training and validation accuracy
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot the training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Create data augmentation layers
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Display augmented images for visualization
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(2):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

# Build a new model with data augmentation and dropout for regularization
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),  # Dropout layer for regularization
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
])

# Compile the new model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Train the model with data augmentation and dropout
epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Extract and plot the new training and validation accuracy/loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Update the range to match the number of epochs in the second training session
epochs_range = range(epochs)  # This should now correspond to 15 epochs

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Load and preprocess a new image for prediction
img = tf.keras.utils.load_img("dog_123.jpg", target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Expand dimensions for batch size

# Make predictions and output the result
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

#TEST
# Load and preprocess a new image for prediction
img = tf.keras.utils.load_img("Cancer (1200).jpg", target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Expand dimensions for batch size

# Make predictions and output the result
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

#TEST
# Load and preprocess a new image for prediction
img = tf.keras.utils.load_img("Not Cancer  (1660).jpg", target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Expand dimensions for batch size

# Make predictions and output the result
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)