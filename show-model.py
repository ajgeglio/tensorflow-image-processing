import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

def parse_args():
    parser = argparse.ArgumentParser(description="Show Keras model summary and plot.")
    parser.add_argument("--model_path", type=str, default="my_model.h5", help="Path to the Keras model file")
    return parser.parse_args()

def main():
    args = parse_args()
    model = load_model(args.model_path)
    name = model.name
    print(model.summary())
    plot_model(model, to_file=f'{name}.png', show_shapes=True, show_layer_names=True)

if __name__ == "__main__":
    main()