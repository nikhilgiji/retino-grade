import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np

class MacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator):
        super().__init__()
        self.val_generator = val_generator

    def on_epoch_end(self, epoch, logs=None):
        val_preds = []
        val_trues = []
        for i in range(len(self.val_generator)):
            x_val, y_val = self.val_generator[i]
            preds = self.model.predict(x_val)
            val_preds.extend(np.argmax(preds, axis=1))
            val_trues.extend(np.argmax(y_val, axis=1))
        f1 = f1_score(val_trues, val_preds, average='macro')
        print(f"\nEpoch {epoch+1} Macro F1: {f1:.4f}")