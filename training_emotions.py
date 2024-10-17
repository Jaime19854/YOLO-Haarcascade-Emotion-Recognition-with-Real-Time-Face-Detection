import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
# Check if a GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Select the first GPU device (you can adjust this as needed)
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(logical_gpus), "Logical GPUs are available.")
    except RuntimeError as e:
        # Error handling if GPU cannot be configured
        print(e)

# Define the path to your dataset
train_dir = 'C:/Users/shay/Desktop/fh/emotions/train'
val_dir = 'C:/Users/shay/Desktop/fh/emotions/test'

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  # Increased for better augmentation
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=32,  # Increased batch size for faster training
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)

# Define the CNN model
no_of_classes = 7

model = Sequential()

# 1st CNN layer
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))  # Increased dropout

# 2nd CNN layer
model.add(Conv2D(128, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# 3rd CNN layer
model.add(Conv2D(256, (3, 3), padding='same'))  # Reduced filters to reduce complexity
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# 4th CNN layer
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# 5th CNN layer
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# Flatten layer
model.add(Flatten())

# Fully connected 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))  # Increased dropout for regularization

# Fully connected 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

# Fully connected 3nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))


# Output layer
model.add(Dense(no_of_classes, activation='softmax'))

# Callbacks for optimization
checkpoint = ModelCheckpoint("./be8t_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)

# Compile the model with an adaptive learning rate optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),  # Can adjust learning rate here
              metrics=['accuracy'])

# Train the model with all the callbacks
# Print the class indices to verify the order of labels
print("Class indices (label mapping):", train_generator.class_indices)

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=100,  # Fewer epochs due to early stopping
                    callbacks=[checkpoint, early_stopping, reduce_lr],
                    verbose=1)
