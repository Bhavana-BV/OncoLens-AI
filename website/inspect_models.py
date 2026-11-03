import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications.resnet50 import preprocess_input

ROOT = Path(r"C:\Users\Bhavana BV\Downloads\OncoLens")

MODEL_PATHS = {
    "MobileNetV2": ROOT / "checkpoints" / "mobilenetv2_sharpen.keras",
    "EfficientNetB0": ROOT / "checkpoints" / "efficientnetb0_sharpen.keras",
    "ResNet50": ROOT / "results" / "resnet50_experiment" / "checkpoints" / "resnet50_best.keras"
}

for name, path in MODEL_PATHS.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    custom_objects = (
        {"preprocess_input": preprocess_input}
        if name == "ResNet50"
        else None
    )

    model = tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects=custom_objects
    )

    print("\nINPUT:", model.input_shape)
    print("OUTPUT:", model.output_shape)

    print("\nTOP-LEVEL LAYERS:")

    for i, layer in enumerate(model.layers):
        print(i, layer.name, type(layer).__name__)

        if isinstance(layer, tf.keras.Model):
            print("  NESTED BACKBONE:")
            for j, sublayer in enumerate(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    print("   ", j, sublayer.name)

    print("\nINSPECTION COMPLETE")

print("\nAll three models inspected.")