<h2>TensorFlow-FlexUNet-Image-Segmentation-Soft-Tissue-Sarcomas (2026/01/19)</h2>
<h3>Revisiting Aerial Semantic Segmentation (Soft-Tissue-Sarcomas)</h3>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for Soft-Tissue-Sarcomas 
based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) , 
and a 512x512 pixels Upscaled<a href="https://drive.google.com/file/d/1naVbtgWJExGRz3eaSUPztCePQ-6Z8lSH/view?usp=sharing">
<b>Augmented-Soft-Tissue-Sarcomas-CTPET-ImageMask-Dataset.zip</b></a>, which was derived by us from
<a href="https://www.kaggle.com/datasets/4quant/soft-tissue-sarcoma/data">
<b>Segmenting Soft Tissue Sarcomas</b></a> on the kaggle web site.
<br><br>
<b>Data Augmentation Strategy</b><br>
To address the limited size of images and masks of the original 
<a href="https://www.kaggle.com/datasets/4quant/soft-tissue-sarcoma/data">
<b>Segmenting Soft Tissue Sarcomas</b></a>,
we used our offline augmentation tool 

<a href="./generator/ImageMaskDatasetGenerator.py"> 
ImageMaskDatasetGenerator.py</a> (see also <a href="https://github.com/sarah-antillia/Image-Distortion-Tool">Image-Distortion-Tool</a>)

to generate the Augmented dataset from <b>lab_petct_vox_5.00mm.h5</b> file in the original dataset.
<br><br> 
<hr>
<b>Actual Image Segmentation for Soft-Tissue-Sarcomas Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks, but they lack precision in certain areas. <br><br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/images/distorted_0.01_rsigma0.5_sigma40_STS_012_32.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/masks/distorted_0.01_rsigma0.5_sigma40_STS_012_32.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test_output/distorted_0.01_rsigma0.5_sigma40_STS_012_32.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/images/distorted_0.02_rsigma0.5_sigma40_STS_005_110.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/masks/distorted_0.02_rsigma0.5_sigma40_STS_005_110.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test_output/distorted_0.02_rsigma0.5_sigma40_STS_005_110.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/images/hflipped_STS_005_110.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/masks/hflipped_STS_005_110.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test_output/hflipped_STS_005_110.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from:<br><br>
<a href="https://www.kaggle.com/datasets/4quant/soft-tissue-sarcoma/data">
<b>Segmenting Soft Tissue Sarcomas</b></a> on the kaggle web site.
<br><br>
<b>About Dataset</b><br>
<b>Summary</b><br>
Summary
The data is a preprocessed subset of the TCIA Study named Soft Tissue Sarcoma. The data have been converted from DICOM folders of varying resolution 
and data types to 3D HDF5 arrays with isotropic voxel size. <br>
This should make it easier to get started and test out various approaches (NN, RF, CRF, etc) to improve segmentations.
<br><br>
<b>TCIA Summary</b><br>
This collection contains FDG-PET/CT and anatomical MR (T1-weighted, T2-weighted with fat-suppression) imaging data from 51 patients with histologically proven soft-tissue sarcomas (STSs) of the extremities. All patients had pre-treatment FDG-PET/CT and MRI scans between November 2004 and November 2011. (Note: date in the TCIA images have been changed in the interest of de-identification; the same change was applied across all images, preserving the time intervals between serial scans). During the follow-up period, 19 patients developed lung metastases. Imaging data and lung metastases development status were used in the following study:

<br><br>
Vallières, M. et al. (2015). A radiomics model from joint FDG-PET and MRI texture features for the prediction of lung metastases in soft-tissue sarcomas of the extremities. Physics in Medicine and Biology, 60(14), 5471-5496. doi:10.1088/0031-9155/60/14/5471.
<br><br>
Imaging data, tumor contours (RTstruct DICOM objects), clinical data and source code is available for this study. See the DOI below for more details and links to access the whole dataset. Please contact Martin Vallières (mart.vallieres@gmail.com) of the Medical Physics Unit of McGill University for any scientific inquiries about this dataset.
<br><br>
<b>Data Citation</b><br>
Vallières, Martin, Freeman, Carolyn R., Skamene, Sonia R., & El Naqa, Issam. (2015). <br>
A radiomics model from joint FDG-PET and MRI texture features for the prediction of lung 
metastases in soft-tissue sarcomas of the extremities.<br>
 The Cancer Imaging Archive. 
 <a href="  http://doi.org/10.7937/K9/TCIA.2015.7GO2GSKS">
 http://doi.org/10.7937/K9/TCIA.2015.7GO2GSKS
</a>
<br><br>
<b>Publication Citation</b><br>
Vallières, M., Freeman, C. R., Skamene, S. R., & Naqa, I. El. (2015, June 29). 
A radiomics model from joint FDG-PET and MRI texture features for the prediction of lung metastases
 in soft-tissue sarcomas of the extremities. <br>
 Physics in Medicine and Biology. IOP Publishing.
 <a href="http://doi.org/10.1088/0031-9155/60/14/5471"> http://doi.org/10.1088/0031-9155/60/14/5471
 </a>
<br><br>
<b>TCIA Citation</b><br>
Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, 
Prior F. <br>
The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, <br>
Volume 26, Number 6, December, 2013, pp 1045-1057. (paper)
<br><br>
<b>License</b><br>
This collection is freely available to browse, download, and use for commercial, scientific and educational purposes <br>
as outlined in the 
<a href="https://spdx.org/licenses/preview/CC-BY-3.0.html">Creative Commons Attribution 3.0 Unported License.</a>
 <br>
See TCIA's Data Usage Policies and Restrictions for additional details. Questions may be directed to help@cancerimagingarchive.net.
<br>
<br>
<h3>
2 Soft-Tissue-Sarcomas ImageMask Dataset
</h3>
 If you would like to train this Soft-Tissue-Sarcomas Segmentation model by yourself,
 please download the original dataset from the google drive  
<a href="https://drive.google.com/file/d/1naVbtgWJExGRz3eaSUPztCePQ-6Z8lSH/view?usp=sharing">
<b>Augmented-Soft-Tissue-Sarcomas-CTPET-ImageMask-Dataset.zip</b></a>
, expand the downloaded, and put it under <b>./dataset </b> folder to be:<br>
<pre>
./dataset
└─Soft-Tissue-Sarcomas
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Soft-Tissue-Sarcomas Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/Soft-Tissue-Sarcomas_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained Soft-Tissue-Sarcomas TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False

num_classes    = 2

base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8

dropout_rate   = 0.04
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for Soft-Tissue-Sarcomas 1+1 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;Soft-Tissue-Sarcomas 1+1
;                      yellow
rgb_map={(0,0,0):0, (255, 255, 0):1,}
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>
By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (23,24,25)</b><br>
<img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (48,49,50)</b><br>
<img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was terminated at epoch 50.<br><br>
<img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/asset/train_console_output_at_epoch50.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for Soft-Tissue-Sarcomas.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/asset/evaluate_console_output_at_epoch50.png" width="880" height="auto">
<br><br>Image-Segmentation-Soft-Tissue-Sarcomas
<a href="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Soft-Tissue-Sarcomas/test was very low, and dice_coef_multiclass veryhigh as shown below.
<br>
<pre>
categorical_crossentropy,0.0052
dice_coef_multiclass,0.9977
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for Soft-Tissue-Sarcomas.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Soft-Tissue-Sarcomas Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the dataset appear similar to the ground truth masks.
<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/images/distorted_0.01_rsigma0.5_sigma40_STS_003_19.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/masks/distorted_0.01_rsigma0.5_sigma40_STS_003_19.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test_output/distorted_0.01_rsigma0.5_sigma40_STS_003_19.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/images/distorted_0.01_rsigma0.5_sigma40_STS_023_133.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/masks/distorted_0.01_rsigma0.5_sigma40_STS_023_133.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test_output/distorted_0.01_rsigma0.5_sigma40_STS_023_133.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/images/distorted_0.01_rsigma0.5_sigma40_STS_023_144.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/masks/distorted_0.01_rsigma0.5_sigma40_STS_023_144.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test_output/distorted_0.01_rsigma0.5_sigma40_STS_023_144.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/images/distorted_0.02_rsigma0.5_sigma40_STS_005_110.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/masks/distorted_0.02_rsigma0.5_sigma40_STS_005_110.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test_output/distorted_0.02_rsigma0.5_sigma40_STS_005_110.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/images/distorted_0.03_rsigma0.5_sigma40_STS_021_155.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/masks/distorted_0.03_rsigma0.5_sigma40_STS_021_155.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test_output/distorted_0.03_rsigma0.5_sigma40_STS_021_155.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/images/hflipped_STS_005_122.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test/masks/hflipped_STS_005_122.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Soft-Tissue-Sarcomas/mini_test_output/hflipped_STS_005_122.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Segmentation model of soft tissue sarcoma based on self-supervised learning</b><br>
Minting Zheng, Chenhua Guo, Yifeng Zhu,  Xiaoming Gang, Chongyang Fu, Shaowu Wang<br>
<a href="https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2024.1247396/full">
https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2024.1247396/full</a>
<br><br>
<b>2. Soft Tissue Sarcoma Co-Segmentation in Combined MRI and PET/CT Data</b><br>
Theresa Neubauer, Maria Wimmer, Astrid Berg, David Major, Dimitrios<br>
Lenis, Thomas Beyer, Jelena Saponjski, and Katja B¨uhler<br>
<a href="https://arxiv.org/pdf/2008.12544">
https://arxiv.org/pdf/2008.12544
</a>
<br>
<br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
