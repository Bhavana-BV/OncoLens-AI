from pathlib import Path
import tensorflow as tf

ROOT = Path(r"C:\Users\Bhavana BV\Downloads\OncoLens")

MODEL_PATHS = {
    "MobileNetV2": ROOT / "checkpoints" / "mobilenetv2_sharpen.keras",
    "EfficientNetB0": ROOT / "checkpoints" / "efficientnetb0_sharpen.keras",
    "ResNet50": ROOT / "results" / "resnet50_experiment" / "checkpoints" / "resnet50_best.keras"
}


def inspect_layers(model, prefix=""):
    results = []

    for layer in model.layers:
        layer_path = f"{prefix}/{layer.name}" if prefix else layer.name

        if isinstance(layer, tf.keras.layers.Conv2D):
            results.append(layer_path)

        if isinstance(layer, tf.keras.Model):
            results.extend(inspect_layers(layer, layer_path))

    return results


for name, path in MODEL_PATHS.items():

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    if name == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input

        model = tf.keras.models.load_model(
            str(path),
            custom_objects={"preprocess_input": preprocess_input},
            compile=False
        )
    else:
        model = tf.keras.models.load_model(
            str(path),
            compile=False
        )

    print("Input:", model.input_shape)
    print("Output:", model.output_shape)

    conv_layers = inspect_layers(model)

    print("\nTotal convolutional layers:", len(conv_layers))
    print("\nLast 15 convolutional layers:")

    for layer_name in conv_layers[-15:]:
        print(layer_name)

    if conv_layers:
        print("\nCandidate final convolutional layer:")
        print(conv_layers[-1])
    else:
        print("\nNo convolutional layers found.")