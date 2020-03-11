import tensorflow as tf
import numpy as np

def MAE(preds, labels):
    MAE = tf.reduce_mean(tf.abs(preds-labels))
    return MAE

def RMSE(preds, labels):
    RMSE = tf.sqrt(tf.reduce_mean(tf.square(preds-labels)))
    return RMSE

def MAPE(preds, labels, scaler, mask_value=None):
    #mean absolute percentage error
    if mask_value != None:
        mask = tf.greater(labels*scaler, mask_value-0.1)
        masked_labels = tf.boolean_mask(labels, mask)
        masked_preds = tf.boolean_mask(preds, mask)
        MAPE = tf.reduce_mean(tf.abs(tf.divide((masked_labels-masked_preds), masked_labels)))
    else:
        MAPE = tf.reduce_mean(tf.abs(tf.divide((labels-preds), labels)))
    return MAPE

def MARE(preds, labels, scaler, mask_value=None):
    #mean absolute relative error
    if mask_value != None:
        mask = tf.greater(labels*scaler, mask_value - 0.1)
        masked_labels = tf.boolean_mask(labels, mask)
        masked_preds = tf.boolean_mask(preds, mask)
        MARE = tf.reduce_sum(tf.abs(masked_labels - masked_preds)) / tf.reduce_sum(masked_labels)
    else:
        MARE = tf.reduce_sum(tf.abs(labels - preds)) / tf.reduce_sum(labels)
    return MARE

def R2(preds, labels):
    mean = tf.reduce_mean(labels)
    r2 = 1 - tf.divide(tf.reduce_sum(tf.square(labels-preds)), tf.reduce_sum(tf.square(labels-mean)))
    return r2

def MAE_NP(preds, labels):
    MAE = np.mean(np.absolute(preds-labels))
    return MAE

def RMSE_NP(preds, labels):
    RMSE = np.sqrt(np.mean(np.square(preds-labels)))
    return RMSE

def MAPE_NP(preds, labels, scaler, mask_value=None):
    if mask_value != None:
        mask = np.where(labels*scaler > (mask_value-0.1), True, False)
        masked_labels = labels[mask]
        masked_preds = preds[mask]
        MAPE = np.mean(np.absolute(np.divide((masked_labels - masked_preds), masked_labels)))
    else:
        MAPE = np.mean(np.absolute(np.divide((labels - preds), labels)))
    return MAPE

def MARE_NP(preds, labels, scaler, mask_value=None):
    if mask_value != None:
        mask = np.where(labels * scaler > (mask_value - 0.1), True, False)
        masked_labels = labels[mask]
        masked_preds = preds[mask]
        MARE = np.divide(np.sum(np.absolute((masked_labels - masked_preds))), np.sum(masked_labels))
    else:
        MARE = np.divide(np.sum(np.absolute((labels - preds))), np.sum(labels))
    return MARE

def R2_NP(preds, labels):
    pass