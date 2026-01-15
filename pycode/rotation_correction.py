import cv2
import numpy as np
import os
import sys

def calculate_rotation_angle(image):
    # Apply edge detection (Canny)
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Perform Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is None:
        return 0  # No lines detected, no rotation needed

    angles_im = []
    for line in lines:
        for rho, theta in line:
            # Convert theta from radians to degrees
            angle = np.degrees(theta)
            # Adjust the angle to the range [-90, 90]
            adjusted_angle = angle - 90 if angle > 90 else angle
            if -10 <= adjusted_angle <= 10:
                angles_im.append(adjusted_angle)

    # plot the images and lines for debug 
    # output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # for line in lines:
    #     for rho, theta in line:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         x1 = int(x0 + 1000 * (-b))
    #         y1 = int(y0 + 1000 * (a))
    #         x2 = int(x0 - 1000 * (-b))
    #         y2 = int(y0 - 1000 * (a))
    #         cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # # Show the image
    # cv2.imshow("Detected Lines", output_image)
    # cv2.waitKey(0)

    # Calculate the average angle
    angles_im = np.array(angles_im)
    
    return angles_im

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

def process_images(image_folder):
    
    # Get all image filenames in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]

    # Calculate the rotation angle for the selected images
    angles = []
    for filename in image_files:
        image_path = os.path.join(image_folder, filename)

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image {filename}")
            continue

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate the rotation angle
        angles_im = calculate_rotation_angle(gray_image)
        for value in angles_im:
            angles.append(value)
       
        # print(f"Rotation angle for {filename}: {angle:.2f} degrees")
    return angles
    # Calculate the average rotation angle
def correct_images(image_folder, angles ):
    if angles:
        # print (angles)
        average_angle = np.mean(angles)
        print(f"Average rotation angle: {average_angle:.2f} degrees")
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        # Correct all images in the folder based on the average angle

        for filename in image_files:
            image_path = os.path.join(image_folder, filename)

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to load image {filename}")
                continue

            # Rotate the image
            corrected_image = rotate_image(image, average_angle)
            output_folder = os.path.join(image_folder, "corrected_images")
            os.makedirs(output_folder, exist_ok=True)

            # Save the corrected image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, corrected_image)

            print(f"Corrected and saved: {filename}")
    else:
        print("No valid rotation angles calculated. Images were not corrected.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    image_folder = sys.argv[1]
    if not os.path.isdir(image_folder):
        print(f"Error: {image_folder} is not a valid directory.")
        sys.exit(1)

    angles = process_images(image_folder)
    correct_images(image_folder,angles)

if __name__ == "__main__":
    main()
