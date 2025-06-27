import tensorflow as tf
import tensorflow_io as tfio

def load_tiff_image(image_path, num_channels):
    # Read the raw bytes from the TIFF file
    file_contents = tf.io.read_file(image_path)
    # Decode the TIFF image
    tensor = tfio.experimental.image.decode_tiff(file_contents)
    # Ensure the tensor has the desired number of channels
    print("Original tensor shape:", tensor.shape)
    tensor = tensor[..., :3]  # Keep only the first 3 channels (RGB) if it has more than 3
    if tensor.shape[-1] != num_channels:
        if num_channels == 1:
            with tf.device('/CPU:0'):
                tensor = tf.image.rgb_to_grayscale(tensor)
        elif num_channels == 3:
            with tf.device('/CPU:0'):
                tensor = tf.image.grayscale_to_rgb(tensor)
        else:
            raise ValueError("Unsupported number of channels requested.")
    return tensor

def main():
    img_size = (1306, 2458)
    num_channels = 1
    image_path = r"Z:\__Organized_Directories_InProgress\2024_UnpackedCollects\20240709_001_REMUS03243_2G\PrimaryImages\PI_1720567332_001_REMUS03243.tif"
    
    # Automatically uses GPU if available, else CPU
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    with tf.device(device):
        image_tensor = load_tiff_image(image_path, num_channels)
        print("Tensor shape:", image_tensor.shape)
        # Optionally resize to img_size if needed
        # image_tensor = tf.image.resize(image_tensor, img_size)

if __name__ == "__main__":
    main()
