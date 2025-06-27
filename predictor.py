import os
import sys
import time
import pandas as pd
import numpy as np
import glob
import tensorflow as tf
from dataloader import DataLoader

class Predict:
    def __init__(
        self,
        name="substrate_inference",
        images_list=None,
        model_dir=None,
        weights_dir="models",
        img_size=(1306, 2458),
        cdictionary=None,
        batch_size=32,
        start_batch=0,
        columns=None,
        save_local_log=True
    ):
        self.name = name
        self.images_list = images_list or []
        self.model_dir = model_dir
        self.weights_dir = weights_dir
        self.img_size = img_size
        self.cdictionary = cdictionary or {0: "coarse", 1: "fine"}
        self.batch_size = batch_size
        self.start_batch = start_batch
        self.columns = columns or ["Filename", "substrate_int", "substrate_class", "image_path"]
        self.save_local_log = save_local_log
        self.dataloader = DataLoader(img_size=self.img_size, num_channels=3)
        self._prepare_output_dir()
    
    def _prepare_output_dir(self):
        output_dir = os.path.join("output", self.name)
        os.makedirs(output_dir, exist_ok=True)

    def generate_substrate_pred(self):
        current_time = time.localtime()
        name_time = time.strftime("%Y-%m-%d-%H-%M", current_time)
        local_log_fname = os.path.join("output", f"{self.name}", f"substrate_prediction_{name_time}.log")
        if self.save_local_log:
            old_stdout = sys.stdout
            log_file = open(local_log_fname, 'w')
            sys.stdout = log_file

        n_imgs = int(len(self.images_list))
        print(f"Performing classification inference on {n_imgs} images")
        df = pd.DataFrame(columns=self.columns)
        prediction_csv = os.path.join("output", f"{self.name}", "substrate_predictions.csv")
        try:
            df.to_csv(prediction_csv, mode='x', header=True)
        except Exception:
            pass

        model = tf.keras.models.load_model(self.weights_dir, compile=False)
        model.load_weights(filepath=self.weights_dir)

        # Prediction Loop
        for k in range(self.start_batch, (n_imgs - 1) // self.batch_size + 1):
            print('batch =', k)
            s = k * self.batch_size
            e = s + self.batch_size
            img_pths = self.images_list[s:e]
            ds = self.dataloader.create_dataset_nl(img_pths)
            img_nms = [os.path.basename(f).split(".")[0] for f in img_pths]
            print(img_nms)
            pred_prob = model.predict(ds)
            pred_int = tf.argmax(pred_prob, axis=-1)
            pred_class = [self.cdictionary[k] for k in pred_int.numpy()]
            print("Class counts: ", np.unique(pred_class, return_counts=True))
            ar = np.c_[img_nms, pred_int, pred_class, img_pths]
            df = pd.DataFrame(ar)
            df.to_csv(prediction_csv, mode='a', header=False)

        # Clean up dataframe at the end
        print("Finished all batches ")
        df = pd.read_csv(prediction_csv, index_col=0)
        df = df.drop_duplicates()
        df.to_csv(prediction_csv, header=True)
        print("Cleaned final predictions and saved csv: ", str(prediction_csv))

        if self.save_local_log:
            sys.stdout = old_stdout
            log_file.close()

        return df

    @staticmethod
    def from_class_type(
        img_pth_lst,
        classes="2c",
        start_batch=0,
        name="substrate_inference",
        img_size=(1306, 2458),
        batch_size=32,
        save_local_log=True,
        weights_dir=None  # Change default to None
    ):
        # Set default weights_dir if not provided
        if weights_dir is None:
            weights_dir = "src\\models"
        if classes == "2c":
            cdictionary = {0: "coarse", 1: "fine"}
            # Match any .h5 file starting with "2class" (fix for missing .h5 in pattern)
            # Try matching any file starting with "2class" if .h5 pattern fails
            pattern = os.path.join(weights_dir, "2class*.h5")
            weights_list = glob.glob(rf"{pattern}")
            if not weights_list:
                raise FileNotFoundError(f"No weights file found matching pattern: {pattern}")
            weights = weights_list[0]
            columns = ["Filename", "substrate_int_2c", "substrate_class_2c", "image_path"]
        elif classes == "6c":
            cdictionary = {
                0: 'consolidated',
                1: 'very_coarse',
                2: 'moderately_coarse',
                3: 'mixed_coarse',
                4: 'mixed',
                5: 'fine'
            }
            pattern = os.path.join(weights_dir, "6class*.h5")
            weights_list = glob.glob(rf"{pattern}")
            weights = weights_list[0]
            if not weights_list:
                raise FileNotFoundError(f"No weights file found matching pattern: {pattern}")
            columns = ["Filename", "substrate_int_6c", "substrate_class_6c", "image_path"]
        else:
            raise ValueError("Unsupported class type. Use '2c' or '6c'.")

        return Predict(
            name=name,
            images_list=img_pth_lst,
            img_size=img_size,
            cdictionary=cdictionary,
            batch_size=batch_size,
            start_batch=start_batch,
            model_dir=None,
            weights_dir=weights,
            columns=columns,
            save_local_log=save_local_log
        )

# Example usage:
# predictor = Predict.from_class_type(img_pth_lst, classes="2c")
# df = predictor.generate_substrate_pred()