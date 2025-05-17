import argparse
import tensorflow as tf
from autokeras.keras_layers import CastToFloat32

def find_model_file(folder_path, model_name):
    """
    Search for the model file in the specified folder path.
    """
    import os

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == model_name:
                return os.path.join(root, file)
            
    return None

def main(folder_path, model_name, save_to_file):
    model = None
    try:
        file_path = find_model_file(folder_path, model_name)
        if file_path:
            # Load the model from the specified file path
            model = tf.keras.models.load_model(file_path)
        else:
            raise FileNotFoundError(f"Model file '{model_name}' not found in '{folder_path}'")
        
        # Visualize the model architecture
        model.summary()

        if save_to_file:
            try:
                import visualkeras
                model_name_without_ending = model_name.split(".")[0]
                # may need to pip install pydot and graphviz
                tf.keras.utils.plot_model(
                    model=model,
                    to_file=f"{folder_path}{model_name_without_ending}_layer_info.png",
                    show_shapes=True,
                    show_layer_names=True,
                    rankdir="TB",
                    expand_nested=True,
                    )
                visualkeras.layered_view(
                    model, 
                    f"{folder_path}{model_name_without_ending}_arch_visualization.png",
                    legend=True,
                    ).show()
            except Exception as e:
                print(f"Error saving model architecture to file: {e}")
                return
    except Exception as e:
        print(f"{e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_path",
        default="./autokeras/breastmnist/",
        help="folder path to search for the model",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        default="breastmnist_autokeras_model1.keras",
        help="name of the pretrained model to test",
        type=str,
    )
    parser.add_argument(
        "--save_to_file",
        default=True,
        help="save the model architecture to a file",
        type=str,
    )
    args = parser.parse_args()
    folder_path = args.folder_path
    model_name = args.model_name
    save_to_file = args.save_to_file
    main(folder_path, model_name, save_to_file)