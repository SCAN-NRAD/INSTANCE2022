#Based on https://raw.githubusercontent.com/ezequieldlrosa/isles22/dad3c35a70dd29e6caae90e730654e5983eb8f97/utils/eval_utils.py
import numpy as np
import cc3d
import nibabel as nb
from glob import glob
import torch
from natsort import natsorted
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff 
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
from numpy import std, mean
from torch.utils.data import DataLoader
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_dice(im1, im2, empty_value=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as im1. If not boolean, it will be converted.
    empty_value : scalar, float.

    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        If both images are empty (sum equal to zero) = empty_value

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.

    This function has been adapted from the Verse Challenge repository:
    https://github.com/anjany/verse/blob/main/utils/eval_utilities.py
    """

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_value

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2.0 * intersection.sum() / im_sum


def compute_absolute_volume_difference(im1, im2, voxel_size):
    """
    Computes the absolute volume difference between two masks.

    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    voxel_size : scalar, float (ml)
        If not float, it will be converted.

    Returns
    -------
    abs_vol_diff : float, measured in ml.
        Absolute volume difference as a float.
        Maximum similarity = 0
        No similarity = inf


    Notes
    -----
    The order of inputs is irrelevant. The result will be identical if `im1` and `im2` are switched.
    """

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    voxel_size = voxel_size.astype(np.float)

    if im1.shape != im2.shape:
        warnings.warn(
            "Shape mismatch: ground_truth and prediction have difference shapes."
            " The absolute volume difference is computed with mismatching shape masks"
        )

    ground_truth_volume = np.sum(im1) * voxel_size
    prediction_volume = np.sum(im2) * voxel_size
    abs_vol_diff = np.abs(ground_truth_volume - prediction_volume)

    return abs_vol_diff


def compute_absolute_lesion_difference(ground_truth, prediction):
    """
    Computes the absolute lesion difference between two masks. The number of lesions are counted for
    each volume, and their absolute difference is computed.

    Parameters
    ----------
    ground_truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.

    Returns
    -------
    abs_les_diff : int
        Absolute lesion difference as integer.
        Maximum similarity = 0
        No similarity = inf


    Notes
    -----
    """
    ground_truth = np.asarray(ground_truth).astype(np.bool)
    prediction = np.asarray(prediction).astype(np.bool)

    ground_truth_numb_lesion = compute_number_of_clusters(ground_truth)
    prediction_numb_lesion = compute_number_of_clusters(prediction)
    abs_les_diff = abs(ground_truth_numb_lesion - prediction_numb_lesion)

    return abs_les_diff


def compute_number_of_clusters(im, connectivity=26):
    """
    Computes the number of 3D clusters (connected-components) in the image.

    Parameters
    ----------
    im : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    connectivity : scalar, int

    Returns
    -------
    num_clusters : scalar, int
    """

    labeled_im = cc3d.connected_components(im, connectivity=connectivity)
    num_clusters = labeled_im.max().astype("int16")

    return num_clusters


def compute_lesion_f1_score(ground_truth, prediction, empty_value=1.0, connectivity=26):
    """
    Computes the lesion-wise F1-score between two masks.

    Parameters
    ----------
    ground_truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    empty_value : scalar, float.
    connectivity : scalar, int.

    Returns
    -------
    f1_score : float
        Lesion-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value

    Notes
    -----
    This function computes lesion-wise score by defining true positive lesions (tp), false positive lesions (fp) and
    false negative lesions (fn) using 3D connected-component-analysis.

    tp: 3D connected-component from the ground-truth image that overlaps at least on one voxel with the prediction image.
    fp: 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
    fn: 3d connected-component from the ground-truth image that has no voxel overlapping with the prediction image.
    """
    ground_truth = np.asarray(ground_truth).astype(np.bool)
    prediction = np.asarray(prediction).astype(np.bool)
    tp = 0
    fp = 0
    fn = 0

    # Check if ground-truth connected-components are detected or missed (tp and fn respectively).
    intersection = np.logical_and(ground_truth, prediction)
    labeled_ground_truth, N = cc3d.connected_components(
        ground_truth, connectivity=connectivity, return_N=True
    )

    # Iterate over ground_truth clusters to find tp and fn.
    # tp and fn are only computed if the ground-truth is not empty.
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_ground_truth, binary=True, in_place=True):
            if np.logical_and(binary_cluster_image, intersection).any():
                tp += 1
            else:
                fn += 1

    # iterate over prediction clusters to find fp.
    # fp are only computed if the prediction image is not empty.
    labeled_prediction, N = cc3d.connected_components(
        prediction, connectivity=connectivity, return_N=True
    )
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_prediction, binary=True, in_place=True):
            if not np.logical_and(binary_cluster_image, ground_truth).any():
                fp += 1

    # Define case when both images are empty.
    if tp + fp + fn == 0:
        if compute_number_of_clusters(ground_truth) == 0:
            f1_score = empty_value
    else:
        f1_score = tp / (tp + (fp + fn) / 2)

    return f1_score


def validation_cycle_multimodal(model, valid_paths, val_lbl_paths, cur_epoch, best_metric, best_hausdorff,
 best_precision, best_metric_epoch, options):
    metric = 0
    val_dwi_paths, val_adc_paths = valid_paths[0],valid_paths[1]
    threshold_pred = 0.9
    model.eval()
    metric_values = list()
    test_dicemetric, test_hausdorffmetric= [],[]
    metric_choosen = 'precision'
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean')
    dice_metric.reset()
    max_intensity = 0
    if options['modality'] =='dwi':
        max_intensity = 1500
    elif options['modality'] =='flair':
        max_intensity = 2000
    elif options['modality'] =='adc':
        max_intensity = 3500
    elif options['modality'] == 'dwi_adc':
        max_intensity_dwi = 1500
        max_intensity_adc = 3500
    
    volumes_recall,volumes_precision = [],[]
    for path_dwi, path_adc, path_label in zip(val_dwi_paths, val_adc_paths,val_lbl_paths):            
###############################  
        test_vol_dwi = nb.load(path_dwi)
        test_lbl = nb.load(path_label)
        test_lbl_pxls = test_lbl.get_fdata()
        test_lbl_pxls = np.array(test_lbl_pxls)
        array_dwi = np.copy(test_vol_dwi.get_fdata())
        in_range = max_intensity_dwi - array_dwi.min()
        array_dwi -= array_dwi.min() #Shifting to 0
        array_dwi /= in_range # 
        out_range = max_intensity - array_dwi.min()
        array_dwi *= array_dwi
        array_dwi += array_dwi.min()
###############################
        test_vol_adc = nb.load(path_adc)
        array_adc = np.copy(test_vol_adc.get_fdata())
        in_range_adc = max_intensity_adc - array_adc.min()
        array_adc -= array_adc.min()
        array_adc /= in_range_adc # 
        out_range_adc = max_intensity_adc - array_adc.min()
        array_adc *= out_range_adc
        array_adc += array_adc.min()
###############################
        dices_volume =[]
        hausdorffs_volume = []
        with torch.no_grad():
            flat_labels, flat_predictions = [],[]
            for slice_selected in range(array_adc.shape[-1]):
                inputs_dwi = torch.tensor(array_dwi[np.newaxis, np.newaxis, :,:,slice_selected], dtype=torch.float).to(device)
                inputs_adc = torch.tensor(array_adc[np.newaxis, np.newaxis, :,:,slice_selected], dtype=torch.float).to(device)
                test_tensor = torch.cat((inputs_dwi, inputs_adc), 1)
                #out_test = model(torch.tensor(test_vol_pxls[np.newaxis, np.newaxis, :,:,slice_selected]).to(device))
                out_test = model(test_tensor)
                out_test = out_test.detach().cpu().numpy()
                pred = np.array(out_test[0,0,:,:]>threshold_pred, dtype='uint8')
                dice_metric(torch.tensor(pred[np.newaxis,np.newaxis,:,:]),torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]))                
                flat_label = np.array(test_lbl_pxls[:,:,slice_selected]>0, dtype='uint8').flatten()
                flat_pred = pred.flatten()
                flat_labels.append(flat_label)
                flat_predictions.append(flat_pred)
            test_dicemetric.append(dice_metric.aggregate().item())
            volumes_recall.append(recall_score(np.array(flat_predictions).flatten(), np.array(flat_labels).flatten()))
            volumes_precision.append(precision_score(np.array(flat_predictions).flatten(), np.array(flat_labels).flatten()))
            #if not torch.isinf(hausdorff_metric.aggregate()[0]) and not torch.isnan(hausdorff_metric.aggregate()[0]):
            #    test_hausdorffmetric.append(hausdorff_metric.aggregate()[0])
        # reset the status for next computation round
        dice_metric.reset()
        #hausdorff_metric.reset()
    metric = np.mean(test_dicemetric)
    hausdorff= 100000000#np.mean(test_hausdorffmetric)
    #metric_values.append(metric)
    if metric > best_metric:
        best_metric = metric
        best_metric_epoch = cur_epoch
        torch.save(model.state_dict(), 'models/'+options['modality']+'_'+options['suffix']+'_best_metric_model_segmentation2d_array.pth')
        print("saved new best metric model")
    print(
        "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
            cur_epoch +1, metric, best_metric, best_metric_epoch
        )
    )
    options['writer'].add_scalar("val_mean_dice", metric, cur_epoch)

    if hausdorff < best_hausdorff:
        best_hausdorff = hausdorff
        best_metric_epoch = cur_epoch
        torch.save(model.state_dict(), 'models/'+options['modality']+'_'+options['suffix']+'_best_hausdorff_model_segmentation2d_array.pth')
        print("saved new best metric model")
    print(
        "current epoch: {} current mean Hausdorff: {:.4f} best mean Hausdorff: {:.4f} at epoch {}".format(
            cur_epoch +1, hausdorff, best_hausdorff, best_metric_epoch
        )
    )        
    avg_precision = mean(volumes_precision)
    if avg_precision > best_precision:
        best_precision = avg_precision
        best_metric_epoch = cur_epoch
        torch.save(model.state_dict(), 'models/'+options['modality']+'_'+options['suffix']+'_best_precision_model_segmentation2d_array.pth')
        print("saved new best metric model")
    print(
        "current epoch: {} current mean Precision: {:.4f} best mean Precision: {:.4f} at epoch {}".format(
            cur_epoch +1, avg_precision, best_precision, best_metric_epoch
        )
    )

    options['writer'].add_scalar("val_mean_dice", metric, cur_epoch)
    options['writer'].add_scalar("val_mean_hausdorff", hausdorff, cur_epoch)
    #writer.add_scalar("val_"+metric_choosen,metric, cur_epoch) TODO: Parametrize choosen metric
    return best_metric, best_hausdorff, best_precision, best_metric_epoch


def validation_cycle(model, valid_paths, val_lbl_paths, cur_epoch, best_metric, best_hausdorff, best_precision, best_metric_epoch, options):
    metric = 0
    threshold_pred = 0.9
    model.eval()
    metric_values = list()
    test_dicemetric, test_hausdorffmetric= [],[]
    metric_choosen = 'precision'
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean')
    dice_metric.reset()
    max_intensity = 0
    if options['modality'] =='dwi':
        max_intensity = 1500
    elif options['modality'] =='flair':
        max_intensity = 2000
    elif options['modality'] =='adc':
        max_intensity = 3500
    elif options['modality'] == 'dwi_adc':
        max_intensity_dwi = 1500
        max_intensity_adc = 3500
    
    volumes_recall,volumes_precision = [],[]
    for path_case, path_label in zip(valid_paths,val_lbl_paths):            
        test_vol = nb.load(path_case)
        test_lbl = nb.load(path_label)
        test_vol_pxls = test_vol.get_fdata()
        test_vol_pxls = np.array(test_vol_pxls, dtype = np.float32)
        test_lbl_pxls = test_lbl.get_fdata()
        test_lbl_pxls = np.array(test_lbl_pxls)
        test_vol_pxls = (test_vol_pxls - 0) / (max_intensity - 0) 
        array = np.copy(test_vol.get_fdata())
        in_range = max_intensity - array.min()
        array -= array.min() #Corriendolo a 0
        array /= in_range # 
        out_range = max_intensity - array.min()
        array *= out_range
        array += array.min()

        dices_volume =[]
        hausdorffs_volume = []
        with torch.no_grad():
            flat_labels, flat_predictions = [],[]
            for slice_selected in range(array.shape[-1]):
                test_tensor = torch.tensor(array[np.newaxis, np.newaxis, :,:,slice_selected], dtype=torch.float).to(device)
                #out_test = model(torch.tensor(test_vol_pxls[np.newaxis, np.newaxis, :,:,slice_selected]).to(device))
                out_test = model(test_tensor)
                out_test = out_test.detach().cpu().numpy()
                pred = np.array(out_test[0,0,:,:]>threshold_pred, dtype='uint8')
                dice_metric(torch.tensor(pred[np.newaxis,np.newaxis,:,:]),torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]))                
                #print("pred_sum:", torch.tensor(pred[np.newaxis,np.newaxis,:,:]).flatten().sum())
                #print("labl_sum:",torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]).flatten().sum())
                #hausdorff_metric(torch.tensor(pred[np.newaxis,np.newaxis,:,:]),torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]))
                flat_label = np.array(test_lbl_pxls[:,:,slice_selected]>0, dtype='uint8').flatten()
                flat_pred = pred.flatten()
                flat_labels.append(flat_label)
                flat_predictions.append(flat_pred)
            test_dicemetric.append(dice_metric.aggregate().item())
            volumes_recall.append(recall_score(np.array(flat_predictions).flatten(), np.array(flat_labels).flatten()))
            volumes_precision.append(precision_score(np.array(flat_predictions).flatten(), np.array(flat_labels).flatten()))
            #if not torch.isinf(hausdorff_metric.aggregate()[0]) and not torch.isnan(hausdorff_metric.aggregate()[0]):
            #    test_hausdorffmetric.append(hausdorff_metric.aggregate()[0])
        # reset the status for next computation round
        dice_metric.reset()
        #hausdorff_metric.reset()
    metric = np.mean(test_dicemetric)
    hausdorff= 100000000#np.mean(test_hausdorffmetric)
    #metric_values.append(metric)
    if metric > best_metric:
        best_metric = metric
        best_metric_epoch = cur_epoch
        torch.save(model.state_dict(), 'models/'+options['modality']+'_'+options['suffix']+'_best_metric_model_segmentation2d_array.pth')
        print("saved new best metric model")
    print(
        "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
            cur_epoch +1, metric, best_metric, best_metric_epoch
        )
    )
    options['writer'].add_scalar("val_mean_dice", metric, cur_epoch)

    if hausdorff < best_hausdorff:
        best_hausdorff = hausdorff
        best_metric_epoch = cur_epoch
        torch.save(model.state_dict(), 'models/'+options['modality']+'_'+options['suffix']+'_best_hausdorff_model_segmentation2d_array.pth')
        print("saved new best metric model")
    print(
        "current epoch: {} current mean Hausdorff: {:.4f} best mean Hausdorff: {:.4f} at epoch {}".format(
            cur_epoch +1, hausdorff, best_hausdorff, best_metric_epoch
        )
    )        
    avg_precision = mean(volumes_precision)
    if avg_precision > best_precision:
        best_precision = avg_precision
        best_metric_epoch = cur_epoch
        torch.save(model.state_dict(), 'models/'+options['modality']+'_'+options['suffix']+'_best_precision_model_segmentation2d_array.pth')
        print("saved new best metric model")
    print(
        "current epoch: {} current mean Precision: {:.4f} best mean Precision: {:.4f} at epoch {}".format(
            cur_epoch +1, avg_precision, best_precision, best_metric_epoch
        )
    )

    options['writer'].add_scalar("val_mean_dice", metric, cur_epoch)
    options['writer'].add_scalar("val_mean_hausdorff", hausdorff, cur_epoch)
    #writer.add_scalar("val_"+metric_choosen,metric, cur_epoch) TODO: Parametrize choosen metric
    return best_metric, best_hausdorff, best_precision, best_metric_epoch




def test_cycle(options, model, all_test_paths, all_test_labels, save_predictions=False):                    
    classification_threshold = 0.9 
    print("Loading best validation model weights: ")
    model_path = 'models/'+options['modality']+'_'+options['suffix']+'_best_metric_model_segmentation2d_array.pth'
    print(model_path)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    pred = []
    test_dicemetric = []
    test_hausdorffs = []
    y = []
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean')
    dice_metric.reset()
    max_intensity = 0
    if options['modality'] =='dwi':
        max_intensity = 1500
        image_paths = dwi_paths
    elif options['modality'] =='flair':
        max_intensity = 2000
        image_paths = flair_paths
    elif options['modality'] =='flair':
        max_intensity = 3500
    print("Computing Test Meassures for " +options['modality'])

    volumes_recall, volumes_precision = [],[]
    for path_test_case, path_test_label in zip(all_test_paths,all_test_labels):                    
        test_vol = nb.load(path_test_case)
        test_lbl = nb.load(path_test_label)
        #print(test_vol.shape == test_lbl.shape)
        flat_labels, flat_predictions = [],[]
        
        test_vol_pxls = test_vol.get_fdata()
        test_vol_pxls = np.array(test_vol_pxls, dtype = np.float32)
        test_lbl_pxls = test_lbl.get_fdata()
        test_lbl_pxls = np.array(test_lbl_pxls)

        test_vol_pxls = (test_vol_pxls - 0) / (max_intensity - 0)
        vol_prediction = np.zeros(test_vol_pxls.shape, dtype = np.float32)
        
        array = np.copy(test_vol.get_fdata())
        in_range = max_intensity - array.min()
        array -= array.min() #Corriendolo a 0
        array /= in_range # 
        out_range = max_intensity - array.min()
        array *= out_range
        array += array.min()
        #print(array.shape)

        for slice_selected in range(array.shape[-1]):
            test_tensor = torch.tensor(array[np.newaxis, np.newaxis, :,:,slice_selected], dtype=torch.float).to(device)
            #print(test_tensor.shape)
            out_test = model(test_tensor)
            out_test = out_test.detach().cpu().numpy()            
            pred = np.array(out_test[0,0,:,:]>classification_threshold, dtype='uint8')
            flat_label = np.array(test_lbl_pxls[:,:,slice_selected]>0, dtype='uint8').flatten()
            flat_pred = pred.flatten()
            flat_labels.append(flat_label)
            flat_predictions.append(flat_pred)
            vol_prediction[:,:,slice_selected] = pred

            dice_metric(torch.tensor(pred[np.newaxis,np.newaxis,:,:]),torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]))
            hausdorff_metric(torch.tensor(pred[np.newaxis,np.newaxis,:,:]),torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]))
        volumes_recall.append(recall_score(np.array(flat_predictions).flatten(), np.array(flat_labels).flatten()))
        volumes_precision.append(precision_score(np.array(flat_predictions).flatten(), np.array(flat_labels).flatten()))
        test_dicemetric.append(dice_metric.aggregate().item())
        if not torch.isinf(hausdorff_metric.aggregate()[0]):
            test_hausdorffs.append(hausdorff_metric.aggregate()[0])
        # reset the status for next computation round
        dice_metric.reset()
        hausdorff_metric.reset()
        if save_predictions:
            img = nb.Nifti1Image(vol_prediction, affine=test_vol.affine, header=test_vol.header)
            Path('outputs/'+options['suffix']).mkdir(parents=True, exist_ok=True)
            nb.save(img, 'outputs/'+options['suffix']+'/'+path_test_case.split('/')[-1]+'_pred_'+options['suffix']+'.nii.gz')

    print(options['suffix']+" model DICE for all TEST VOLUMES: ")
    print(np.mean(test_dicemetric))
    return(np.mean(test_dicemetric),np.mean(test_hausdorffs),volumes_precision, volumes_recall)


def test_cycle_multimodal(options, model, all_test_paths, all_test_labels, save_predictions=False):                    
    classification_threshold = 0.9 
    print("Loading best validation model weights: ")    
    model_path = 'models/'+options['modality']+'_'+options['suffix']+'_best_metric_model_segmentation2d_array.pth'
    print(model_path)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    test_dwi_paths, test_adc_paths = all_test_paths[0],all_test_paths[1]

    pred = []
    test_dicemetric = []
    test_hausdorffs = []
    y = []
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean')
    dice_metric.reset()
    max_intensity = 0
    if options['modality'] =='dwi':
        max_intensity = 1500
    elif options['modality'] =='flair':
        max_intensity = 2000
    elif options['modality'] =='flair':
        max_intensity = 3500
    elif options['modality'] == 'dwi_adc':
        max_intensity_dwi = 1500
        max_intensity_adc = 3500
    print("Computing Test Meassures for " +options['modality'])
     
    volumes_recall, volumes_precision = [],[]
    for path_dwi, path_adc, path_label in zip(test_dwi_paths, test_adc_paths,all_test_labels):                    
    ###############################  
        test_vol_dwi = nb.load(path_dwi)
        test_lbl = nb.load(path_label)
        test_lbl_pxls = test_lbl.get_fdata()
        test_lbl_pxls = np.array(test_lbl_pxls)
        array_dwi = np.copy(test_vol_dwi.get_fdata())
        in_range = max_intensity_dwi - array_dwi.min()
        array_dwi -= array_dwi.min() #Shifting to 0
        array_dwi /= in_range # 
        out_range = max_intensity - array_dwi.min()
        array_dwi *= array_dwi
        array_dwi += array_dwi.min()
    ###############################
        test_vol_adc = nb.load(path_adc)
        array_adc = np.copy(test_vol_adc.get_fdata())
        in_range_adc = max_intensity_adc - array_adc.min()
        array_adc -= array_adc.min()
        array_adc /= in_range_adc # 
        out_range_adc = max_intensity_adc - array_adc.min()
        array_adc *= out_range_adc
        array_adc += array_adc.min()
    ###############################
        vol_prediction = np.zeros(array_adc.shape, dtype = np.float32)

        with torch.no_grad():
            flat_labels, flat_predictions = [],[]
            for slice_selected in range(array_adc.shape[-1]):
                inputs_dwi = torch.tensor(array_dwi[np.newaxis, np.newaxis, :,:,slice_selected], dtype=torch.float).to(device)
                inputs_adc = torch.tensor(array_adc[np.newaxis, np.newaxis, :,:,slice_selected], dtype=torch.float).to(device)
                test_tensor = torch.cat((inputs_dwi, inputs_adc), 1)
                out_test = model(test_tensor)
                out_test = out_test.detach().cpu().numpy()            
                pred = np.array(out_test[0,0,:,:]>classification_threshold, dtype='uint8')
                flat_label = np.array(test_lbl_pxls[:,:,slice_selected]>0, dtype='uint8').flatten()
                flat_pred = pred.flatten()
                flat_labels.append(flat_label)
                flat_predictions.append(flat_pred)
                vol_prediction[:,:,slice_selected] = pred

                dice_metric(torch.tensor(pred[np.newaxis,np.newaxis,:,:]),torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]))
                hausdorff_metric(torch.tensor(pred[np.newaxis,np.newaxis,:,:]),torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]))
        volumes_recall.append(recall_score(np.array(flat_predictions).flatten(), np.array(flat_labels).flatten()))
        volumes_precision.append(precision_score(np.array(flat_predictions).flatten(), np.array(flat_labels).flatten()))
        test_dicemetric.append(dice_metric.aggregate().item())
        if not torch.isinf(hausdorff_metric.aggregate()[0]):
            test_hausdorffs.append(hausdorff_metric.aggregate()[0])
        # reset the status for next computation round
        dice_metric.reset()
        hausdorff_metric.reset()
        if save_predictions:
            img = nb.Nifti1Image(vol_prediction, affine=test_vol_dwi.affine, header=test_vol_dwi.header)
            Path('outputs/'+options['suffix']).mkdir(parents=True, exist_ok=True)
            nb.save(img, 'outputs/'+options['suffix']+'/'+path_dwi.split('/')[-1]+'_pred_'+options['suffix']+'.nii.gz')

    print(options['suffix']+" model DICE for all TEST VOLUMES: ")
    print(np.mean(test_dicemetric))
    print("Average precision: " + str(np.mean(volumes_precision)))
    print("Average recall: " + str(np.mean(volumes_recall)))
    return(np.mean(test_dicemetric),np.mean(test_hausdorffs),volumes_precision, volumes_recall)


    """
    Computes the lesion-wise F1-score between two masks.

    Parameters
    ----------
    ground_truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    empty_value : scalar, float.
    connectivity : scalar, int.

    Returns
    -------
    f1_score : float
        Lesion-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value

    Notes
    -----
    This function computes lesion-wise score by defining true positive lesions (tp), false positive lesions (fp) and
    false negative lesions (fn) using 3D connected-component-analysis.

    tp: 3D connected-component from the ground-truth image that overlaps at least on one voxel with the prediction image.
    fp: 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
    fn: 3d connected-component from the ground-truth image that has no voxel overlapping with the prediction image.
    """
    ground_truth = np.asarray(ground_truth).astype(np.bool)
    prediction = np.asarray(prediction).astype(np.bool)
    tp = 0
    fp = 0
    fn = 0

    # Check if ground-truth connected-components are detected or missed (tp and fn respectively).
    intersection = np.logical_and(ground_truth, prediction)
    labeled_ground_truth, N = cc3d.connected_components(
        ground_truth, connectivity=connectivity, return_N=True
    )

    # Iterate over ground_truth clusters to find tp and fn.
    # tp and fn are only computed if the ground-truth is not empty.
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_ground_truth, binary=True, in_place=True):
            if np.logical_and(binary_cluster_image, intersection).any():
                tp += 1
            else:
                fn += 1

    # iterate over prediction clusters to find fp.
    # fp are only computed if the prediction image is not empty.
    labeled_prediction, N = cc3d.connected_components(
        prediction, connectivity=connectivity, return_N=True
    )
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_prediction, binary=True, in_place=True):
            if not np.logical_and(binary_cluster_image, ground_truth).any():
                fp += 1

    # Define case when both images are empty.
    if tp + fp + fn == 0:
        if compute_number_of_clusters(ground_truth) == 0:
            f1_score = empty_value
    else:
        f1_score = tp / (tp + (fp + fn) / 2)

    return f1_score