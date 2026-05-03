import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


#Location for the dataset folder
DATASET_DIR = "./dataset"

# Output folder
OUTPUT_DIR = "./output"

# Image dimensions (MobileNetV2 default input)
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Training parameters
BATCH_SIZE = 32
EPOCHS_TRANSFER = 15      # Epochs for initial transfer learning phase
EPOCHS_FINE_TUNE = 10     # Epochs for fine-tuning phase
LEARNING_RATE = 0.001
FINE_TUNE_LR = 0.0001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Random seed for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

#Creating an outputfile if it doesnt already exist
def create_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

#Splits the data into training and testing data
def load_and_split_data():
    print("\n" + "=" * 50)
    print("LOADING DATASET")
    print("=" * 50)

    # First, load train set (70% of data)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT + TEST_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    # Load the remaining 30% (val + test)
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT + TEST_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    print(f"\nClasses found: {class_names}")
    print(f"Number of classes: {len(class_names)}")

    # Shuffle val_test before splitting to ensure balanced class distribution
    val_test_size = tf.data.experimental.cardinality(val_test_ds).numpy()
    val_test_ds = val_test_ds.unbatch().shuffle(buffer_size=10000, seed=SEED).batch(BATCH_SIZE)
    val_size = val_test_size // 2

    val_ds = val_test_ds.take(val_size)
    test_ds = val_test_ds.skip(val_size)

    # Count images per class for class weight computation
    print("\nCounting images per class...")
    count_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        image_size=IMG_SIZE,
        batch_size=1,
        label_mode="int",
        shuffle=False,
    )
    labels = np.concatenate([y.numpy() for _, y in count_ds])
    for i, name in enumerate(class_names):
        print(f"  {name}: {np.sum(labels == i)} images")

    # Compute class weights to compensate for imbalance
    class_indices = np.arange(len(class_names))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_indices,
        y=labels,
    )
    class_weights = dict(enumerate(weights))
    print("\nClass weights (higher = underrepresented):")
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weights[i]:.3f}")

    print(f"\nDataset split:")
    print(f"  Train batches:      {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"  Validation batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"  Test batches:       {tf.data.experimental.cardinality(test_ds).numpy()}")

    # Optimize performance with prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names, class_weights


def build_model(num_classes):
    print("\n" + "=" * 50)
    print("BUILDING MODEL")
    print("=" * 50)

    # Load MobileNetV2 without the top classification layer
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze base model weights for transfer learning
    base_model.trainable = False

    # Build the complete model
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Data augmentation on raw pixels [0, 255] — active only during training
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.2)(x)
    x = layers.RandomBrightness(0.2)(x)
    x = layers.RandomContrast(0.2)(x)

    # Preprocessing: scale pixel values to [-1, 1] for MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Feature extraction
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    # Compile with Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    print(f"\nBase model layers: {len(base_model.layers)}")
    print(f"Total parameters: {model.count_params():,}")

    return model, base_model


def train_model(model, base_model, train_ds, val_ds, class_weights):
    print("\n" + "=" * 50)
    print("PHASE 1: TRANSFER LEARNING")
    print("=" * 50)

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    # Phase 1: Transfer learning
    history_transfer = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_TRANSFER,
        class_weight=class_weights,
        callbacks=[early_stop],
    )

    # Phase 2: Fine-tuning
    print("\n" + "=" * 50)
    print("PHASE 2: FINE-TUNING")
    print("=" * 50)

    # Unfreeze the top layers of the base model
    base_model.trainable = True

    # Freeze everything except the last 30 layers
    fine_tune_from = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"Fine-tuning {trainable_count} layers of base model")

    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop_ft = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNE,
        class_weight=class_weights,
        callbacks=[early_stop_ft],
    )

    return history_transfer, history_finetune


def evaluate_model(model, test_ds, class_names):
    print("\n" + "=" * 50)
    print("EVALUATING MODEL")
    print("=" * 50)

    # Overall test accuracy
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.1f}%)")

    # Get predictions for detailed metrics
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report
    print("\nClassification Report:")
    print("-" * 50)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # Save report to file
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return test_accuracy, cm, y_true, y_pred


def plot_training_history(history_transfer, history_finetune):
    acc = history_transfer.history["accuracy"] + history_finetune.history["accuracy"]
    val_acc = history_transfer.history["val_accuracy"] + history_finetune.history["val_accuracy"]
    loss = history_transfer.history["loss"] + history_finetune.history["loss"]
    val_loss = history_transfer.history["val_loss"] + history_finetune.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)
    phase1_end = len(history_transfer.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, acc, label="Train Accuracy", linewidth=2)
    ax1.plot(epochs_range, val_acc, label="Val Accuracy", linewidth=2)
    ax1.axvline(x=phase1_end, color="gray", linestyle="--", label="Fine-tuning start")
    ax1.set_title("Model Accuracy", fontsize=14)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, loss, label="Train Loss", linewidth=2)
    ax2.plot(epochs_range, val_loss, label="Val Loss", linewidth=2)
    ax2.axvline(x=phase1_end, color="gray", linestyle="--", label="Fine-tuning start")
    ax2.set_title("Model Loss", fontsize=14)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "training_history.png"), dpi=150)
    plt.show()
    print("Saved: training_history.png")


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "confusion_matrix.png"), dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")


def save_keras_model(model):
    model_path = os.path.join(OUTPUT_DIR, "waste_classifier.keras")
    model.save(model_path)
    print(f"\nKeras model saved: {model_path}")
    return model_path


def convert_to_tflite(model):
    print("\n" + "=" * 50)
    print("CONVERTING TO TENSORFLOW LITE")
    print("=" * 50)

    # Standard TFLite conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_path = os.path.join(OUTPUT_DIR, "waste_classifier.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"Standard TFLite model saved: {tflite_path} ({size_mb:.2f} MB)")

    # Quantized TFLite conversion (optimized for RPi) 
    converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_quant_model = converter_quant.convert()

    tflite_quant_path = os.path.join(OUTPUT_DIR, "waste_classifier_quantized.tflite")
    with open(tflite_quant_path, "wb") as f:
        f.write(tflite_quant_model)

    size_quant_mb = os.path.getsize(tflite_quant_path) / (1024 * 1024)
    print(f"Quantized TFLite model saved: {tflite_quant_path} ({size_quant_mb:.2f} MB)")
    print(f"Size reduction: {(1 - size_quant_mb / size_mb) * 100:.1f}%")

    return tflite_path, tflite_quant_path


def verify_tflite_model(tflite_path, test_ds, class_names):
    print("\n" + "=" * 50)
    print("VERIFYING TFLITE MODEL")
    print("=" * 50)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input shape:  {input_details[0]['shape']}")
    print(f"Input dtype:  {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")

    correct = 0
    total = 0

    for images, labels in test_ds:
        for i in range(images.shape[0]):
            img = np.expand_dims(images[i].numpy(), axis=0).astype(np.float32)

            interpreter.set_tensor(input_details[0]["index"], img)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]["index"])

            pred_class = np.argmax(output)
            true_class = np.argmax(labels[i].numpy())

            if pred_class == true_class:
                correct += 1
            total += 1

    tflite_accuracy = correct / total
    print(f"\nTFLite Model Accuracy: {tflite_accuracy:.4f} ({tflite_accuracy * 100:.1f}%)")
    print(f"Tested on {total} images")

    return tflite_accuracy


def save_class_labels(class_names):
    labels_path = os.path.join(OUTPUT_DIR, "labels.txt")
    with open(labels_path, "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Class labels saved: {labels_path}")





def main():
    print("=" * 50)
    print("WASTE SORTING CLASSIFIER - TRAINING PIPELINE")
    print("=" * 50)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    # Step 1: Setup
    create_output_dir()

    # Step 2: Load data + compute class weights
    train_ds, val_ds, test_ds, class_names, class_weights = load_and_split_data()

    # Step 3: Build model
    model, base_model = build_model(num_classes=len(class_names))

    # Step 4: Train (with class weights)
    history_transfer, history_finetune = train_model(
        model, base_model, train_ds, val_ds, class_weights
    )

    # Step 5: Evaluate
    test_accuracy, cm, y_true, y_pred = evaluate_model(model, test_ds, class_names)

    # Step 6: Plot results
    plot_training_history(history_transfer, history_finetune)
    plot_confusion_matrix(cm, class_names)

    # Step 7: Save Keras model
    save_keras_model(model)

    # Step 8: Convert to TFLite
    tflite_path, tflite_quant_path = convert_to_tflite(model)

    # Step 9: Verify TFLite model
    tflite_accuracy = verify_tflite_model(tflite_path, test_ds, class_names)

    # Step 10: Save labels
    save_class_labels(class_names)

    # Summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 50)
    print(f"Classes:              {class_names}")
    print(f"Keras Test Accuracy:  {test_accuracy:.4f}")
    print(f"TFLite Test Accuracy: {tflite_accuracy:.4f}")
    print(f"\nFiles saved in '{OUTPUT_DIR}/':")
    print(f"  - waste_classifier.keras          (full model)")
    print(f"  - waste_classifier.tflite         (for RPi)")
    print(f"  - waste_classifier_quantized.tflite (smaller, for RPi)")
    print(f"  - labels.txt                      (class names)")
    print(f"  - classification_report.txt       (detailed metrics)")
    print(f"  - plots/training_history.png")
    print(f"  - plots/confusion_matrix.png")


if __name__ == "__main__":
    main()
