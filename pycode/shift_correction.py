import cv2
import numpy as np
import glob
import os
import sys

def compute_shift(image1, image2):
    # Compute normalized cross-correlation
    result = cv2.matchTemplate(image1, image2, cv2.TM_CCORR_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    
    # Compute shift in x-direction
    shift_x = max_loc[0] - (image2.shape[1] // 2)
    return shift_x

def compute_shift_ecc(reference_image, target_image):
    warp_matrix = np.eye(2, 3, dtype=np.float32)  # Identity transformation

    # Convert images to float32
    ref_gray = np.float32(reference_image)
    target_gray = np.float32(target_image)

    # ECC alignment
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
    cc, warp_matrix = cv2.findTransformECC(ref_gray, target_gray, warp_matrix, cv2.MOTION_TRANSLATION, criteria)

    shift_x = warp_matrix[0, 2]  # Horizontal shift
    return int(shift_x)

def shift_correction(path):
    image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        # Correct all images in the folder based on the average angle
    # Load all images from a folder (Example: "*.png" or "*.jpg")
    reference_image = image_files[1]
    ref_image = cv2.imread(path+reference_image)
    # print (ref_image)
    ref_gray_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    for i,filename in enumerate(image_files):
        image_path = os.path.join(path, filename)
        # Read the image
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        shift_x = compute_shift_ecc(ref_gray_image, gray_image)
        print (shift_x, " | ")
        corrected_img = image
        if (shift_x != 0) : 
            corrected_img = np.roll(image, -shift_x, axis=1) 
        output_folder = os.path.join(path, "shifted_images")
        os.makedirs(output_folder, exist_ok=True)

            # Save the corrected image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, corrected_img)
    # Select the reference frame (first frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    image_folder = sys.argv[1]
    if not os.path.isdir(image_folder):
        print(f"Error: {image_folder} is not a valid directory.")
        sys.exit(1)

    
    shift_correction(image_folder)
    

if __name__ == "__main__":
    main()
