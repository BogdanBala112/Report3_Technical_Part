# Report3_Technical_Part

Image rotation correction
Image rotation correction is performed using the software routine rotation_correction.py, supplying the source folder that contains the image set to be processed. The results are stored in the corrected_images folder.
Example:

First of all navigate to the folder containing all the scripts used to rotate those images.
- use this command cd pycode

!!! python3 rotation_correction.py "../MDA231_ELK-KD_R1-P1_E1_26.09.23_P1_10x" !!!
- this represents the folder containing images as they were found in the dataset
- by giving this line "MDA231_ELK-KD_R1-P1_E1_26.09.23_P1_10x" the program will detect the images from this specified folder and it will apply the rotation logic neccesarry to the images to result in a clearer and suitable image to perform the next analysis on it.

Determination of differences between acquisition sequences
Apply morphology and normalization, and then perform contour detection.
The most important step is contour filtering, which is done according to:
size (area)
remove images in which there are very many cell detections across the entire surface, or across 50% of the surface
OR, from these detections, keep only those with higher intensity
remove contours that are wider than a certain distance, determined as part of the rotation step
