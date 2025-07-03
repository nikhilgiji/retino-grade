import os
import logging
import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from src.data_loader import DataLoader, DataLoaderConfig
from src.utils import setup_logger, create_dirs

def inference(config, model_path):
    # Ensure output and log directories exist
    create_dirs([config.output_dir, config.log_dir])
    
    # Setup logger
    logger = setup_logger("inference_logger", os.path.join(config.log_dir, "inference.log"))
    
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    data_loader = DataLoader(config)
    test_generator = data_loader.get_test_generator()
    
    logger.info("Starting predictions on test data")
    test_generator.reset()
    pred_probs = model.predict(test_generator, verbose=1)
    pred_classes = pred_probs.argmax(axis=1)
    
    class_indices = test_generator.class_indices
    inv_class_indices = {v: k for k, v in class_indices.items()}
    pred_labels = [inv_class_indices[i] for i in pred_classes]
    
    # Save predictions CSV
    test_df = data_loader.test_df.copy()
    test_df['predicted_label'] = pred_labels
    pred_csv_path = os.path.join(config.output_dir, 'test_predictions_cbam.csv')
    test_df[['id_code', 'predicted_label']].to_csv(pred_csv_path, index=False)
    logger.info(f"Saved predictions to {pred_csv_path}")
    
    # If true labels available, evaluate and save confusion matrix
    if 'diagnosis' in test_df.columns:
        y_true = test_generator.classes
        logger.info("Generating classification report")
        
        # Classification report as dict & string
        cls_report_dict = classification_report(y_true, pred_classes, target_names=list(class_indices.keys()), output_dict=True)
        cls_report_str = classification_report(y_true, pred_classes, target_names=list(class_indices.keys()))
        
        logger.info("\nClassification Report:\n" + cls_report_str)
        logger.info("Classification Report (dict):\n" + json.dumps(cls_report_dict, indent=2))
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, pred_classes)
        logger.info(f"Cohen's Kappa: {kappa:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        
        # Confusion matrix plot
        cm = confusion_matrix(y_true, pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_indices.keys(),
                    yticklabels=class_indices.keys())
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix on Test Set")
        
        cm_path = os.path.join(config.output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        logger.info(f"Saved confusion matrix to {cm_path}")
        plt.close()