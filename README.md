# Whole Image Classifier

This repository provides tools for running substrate classification inference on underwater images using pre-trained deep learning models. The weights are for a sequential deep CNN classification model developed with Tensorflow.

## Directory Structure

- `src/`  
  Core source code, including:
  - `predictor.py`: Main prediction logic and batch inference.
  - `dataloader.py`: Image loading utilities.
  - `utils.py`: Helper functions.

- `scripts/`  
  Command-line scripts for running inference, e.g. `infer.py`.

- `notebooks/`  
  Jupyter notebooks for interactive experimentation and batch runs.

- `output/`  
  Default output directory for logs and prediction CSVs.

- `models/`  
  Directory for storing model weights (e.g., `.h5` files).

## Usage

### Command Line

Run inference using the provided script:

```sh
python scripts/infer.py --image_dir <IMAGE_DIR> --classes 2c --name <RUN_NAME>
```

**Arguments:**
- `--image_dir`: Directory containing images for inference.
- `--image_list_file`: Text file with image paths (one per line).
- `--classes`: Model type, `"2c"` or `"6c"`.
- `--name`: Name for the prediction run (used for output folder).
- `--start_batch`: Batch index to start from (default: 0).

### In Python

```python
from predictor import Predict

img_pth_lst = [...]  # List of image paths
predictor = Predict.from_class_type(img_pth_lst, classes="2c", name="my_run")
df = predictor.generate_substrate_pred()
```

## Model Weights

Place your model weights (e.g., `2class*.h5` or `6class*.h5`) in the `src/models/` directory.

## Output

Predictions and logs are saved in `output/<RUN_NAME>/`.

## Requirements

- Python 3.8+
- TensorFlow
- pandas, numpy

Install dependencies with:

```sh
pip install -r requirements.txt
```
or
```sh
conda install -f environment.yml
```

## Troubleshooting

- Ensure model weights are present in the correct directory.
- If you encounter import errors, run scripts from the project root or use absolute paths.

## License

GNU GENERAL PUBLIC LICENSE

# Required GPU Libraries for TensorFlow 2.10.0
To enable GPU support, you need to install:
- CUDA Toolkit 11.2
- cuDNN 8.1

These are the officially supported versions for TensorFlow 2.10.0 on Windows.

🛠️ Steps to Set Up GPU Support
- Uninstall any conflicting CUDA/cuDNN versions
Multiple versions can cause DLL conflicts. Keep only:
- NVIDIA Graphics Driver
- NVIDIA Control Panel
- NVIDIA GeForce Experience (optional)
- Install CUDA 11.2
- Download from NVIDIA’s CUDA 11.2 archive.
- During install, choose Custom and deselect unnecessary components like Nsight.
- Install cuDNN 8.1 for CUDA 11.2
- Sign in to the NVIDIA Developer site and download cuDNN 8.1.
- Unzip and copy the contents into your CUDA installation directory:
- bin/ → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
- include/ → ...\include
- lib/x64/ → ...\lib\x64

# Add to Environment Variables
- Add these to your PATH:

```sh
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp
```

# Verify GPU Availability In Python

```sh
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

## or

```sh
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
```

Once that’s in place, TensorFlow should enable access to DLLs and start using your GPU. 
