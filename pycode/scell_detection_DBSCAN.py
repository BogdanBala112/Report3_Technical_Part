import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
from skimage import measure
import pandas as pd
from scipy.interpolate import make_interp_spline
from collections import defaultdict
from sklearn.cluster import DBSCAN


def compute_centroid(contour):
    return np.mean(contour, axis=0)  # Returns (y, x)

def group_contours_by_x_position(data, tolerance=5):
    """
    Groups contours based on their x position range (+/- tolerance pixels),
    while also keeping track of the corresponding index.

    :param data: List of dictionaries containing contour data.
    :param tolerance: Allowed pixel difference for grouping.
    :return: Dictionary where keys are group indices and values are lists of (index, contour) tuples.
    """
    contour_groups = defaultdict(list)
    group_id = 1

    for _, row in data.iterrows():
        for (contour, centroid) in row["contours"]:
            # print ( "contour centroid = ",contour, centroid)
            # Check if the contour belongs to an existing group
            found_group = False
            # print("contour_groups = ", contour_groups)
            for key, group in contour_groups.items():
                if abs(group[0][2][1] - centroid[1]) <= tolerance:
                    print("row[Index] = ", row["Index"])
                    contour_groups[key].append([row["Index"], contour, centroid])
                    found_group = True
                    break

            # If not found, create a new group
            if not found_group:
                contour_groups[group_id].append([row["Index"], contour, centroid])
                group_id += 1
    filtered_contour_groups = {k: v for k, v in contour_groups.items() if len(v) >= 120}
    return dict(filtered_contour_groups)


# Function to merge contours only if their centroids are aligned on Y and are close
def merge_contours_by_alignment(contours, y_threshold, x_tolerance):
    merged_contours = []
    used = set()

    for i, c1 in enumerate(contours):
        if i in used:
            continue
        c1_centroid = compute_centroid(c1)
        # print("c1_centroid " , c1_centroid)
        y_min1, y_max1 = np.min(c1[:, 0]), np.max(c1[:, 0])
        merged_contour = c1.copy()

        for j, c2 in enumerate(contours):
            if i != j and j not in used:
                c2_centroid = compute_centroid(c2)
                # print("c2_centroid " , c2_centroid)
                y_min2, y_max2 = np.min(c2[:, 0]), np.max(c2[:, 0])
                # print("y_min2 = %d, y_max2 = %d", y_min2, y_max2)
                # Check if centroids are aligned on Y-axis (similar X values)
                if abs(c1_centroid[1] - c2_centroid[1]) <= x_tolerance:
                    # Check if contours are close in Y distance
                    if abs(y_min1 - y_max2) < y_threshold or abs(y_max1 - y_min2) < y_threshold:
                        merged_contour = np.vstack([merged_contour, c2])
                        used.add(j)
        cv2.waitKey(0)
        merged_contours.append(merged_contour)
        used.add(i)

    return merged_contours

def shrink_contours_inward(contours, scale_factor=0.9):
    """
    Shrinks contours by moving points toward their centroid.
    
    :param contours: List of contours from skimage.measure.find_contours.
    :param scale_factor: Reduction factor (0.9 means 90% of original size).
    :return: List of reduced contours.
    """
    reduced_contours = []
    
    for contour in contours:
        contour = np.array(contour, dtype=np.float32)  # Ensure correct format
        
        # Compute centroid
        cx = np.mean(contour[:, 1])  # X centroid
        cy = np.mean(contour[:, 0])  # Y centroid
        
        # Move points toward centroid
        new_contour = scale_factor * (contour - [cy, cx]) + [cy, cx]
        
        reduced_contours.append(new_contour.astype(np.float32))

    return reduced_contours

def get_convex_contour(contour):
    if len(contour) < 3:
        return contour  # A convex hull needs at least 3 points

    hull = ConvexHull(contour)  # Compute convex hull
    convex_contour = contour[hull.vertices]  # Extract convex hull points
    convex_contour = np.vstack([convex_contour, convex_contour[0]])
    return convex_contour

def preprocess_contours(contours):
    """
    Converts contour points to integer values and corrects shape for OpenCV.
    
    :param contours: List of raw contours (float values)
    :return: List of formatted contours (integer values)
    """
    processed_contours = []
    for contour in contours:
        if len(contour) > 0:
            contour = np.array(contour, dtype=np.int32)  # Convert float to int
            # contour = contour.reshape((-1, 1, 2))  # Ensure proper shape (N,1,2)
            processed_contours.append(contour)
    return processed_contours

def process_images(image_folder):
  
    diff_folder = os.path.join(image_folder, "diff_images")
    os.makedirs(diff_folder, exist_ok=True)

    # Get all image filenames in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
    def extract_key(filename):
        # Extract the last 5 characters, convert to int for sorting
        return int(filename[-9:-4])  # Adjust slicing if needed
    
    image_files = sorted(image_files, key=extract_key) 
    # print(image_files)   
    data = []
    data_contours = []
    previous_image = None
    
    for index, filename in enumerate(image_files):
        if index == 1:
            image_path = os.path.join(image_folder, filename)
            image_mask = cv2.imread(image_path)
            image_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)

        image_path = os.path.join(image_folder, filename)

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image {filename}")
            continue

        # Calculate the difference between consecutive images and save
        if previous_image is not None:
            diff = cv2.absdiff(previous_image, image)
            if len(diff.shape) == 3:  # Check if the image has multiple channels (i.e., color)
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            else:
                diff_gray = diff  # Already grayscale
            
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # Apply CLAHE
            # norm_image = clahe.apply(diff_gray)
            norm_diff = cv2.normalize(diff_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            gauss_diff = cv2.GaussianBlur(norm_diff, (7, 9), sigmaX=2)
            # -------------- bilateral filtering not used ---------------------
            # filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

#  --------------------- working area ------------------------------------
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
            # Apply dilation
            dilated_grayscale = cv2.dilate(gauss_diff, kernel, iterations=1)
            normalized_image = cv2.normalize(dilated_grayscale, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            image_data = np.array(normalized_image)
            x = np.linspace(0, image_data.shape[1], image_data.shape[1])
            y = np.linspace(0, image_data.shape[0], image_data.shape[0])
            x, y = np.meshgrid(x, y)
            z = image_data
            z_intersect = 45  # De exemplu, un plan la valoarea 128 de gri
 
            # Găsirea punctelor de intersecție
            contours = measure.find_contours(z, z_intersect)
            filtered_contours = []
            min_area = 150
            min_x_extent = 30

            # Calcularea ariei regiunilor de intersecție
            areas = []
            for contour in contours:
                x_contour = contour[:, 1]
                y_contour = contour[:, 0]
                # Folosim formula poligonului pentru a calcula aria
                n = len(x_contour)
                area = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    area += x_contour[i] * y_contour[j]
                    area -= y_contour[i] * x_contour[j]
                area = abs(area) / 2.0
                x_min = x_contour.min()
                x_max = x_contour.max()
                x_extent = x_max - x_min  # Width along the x-axis
                # areas.append(area)
                if area >= min_area and x_extent <= min_x_extent:
                    filtered_contours.append(contour)
                    areas.append(area)

            y_distance_threshold = 20
            x_tolerance = 10
            # Apply merging based on Y-alignment and proximity
            merged_contours = merge_contours_by_alignment(filtered_contours, y_distance_threshold, x_tolerance)

            convex_contours = [get_convex_contour(contour) for contour in merged_contours]
            filtered_contours  = convex_contours
            shrunken_contours = shrink_contours_inward(filtered_contours,0.85)
            contours_cnt = []
            for contour in shrunken_contours:
                contours_cnt.append((contour, compute_centroid (contour) ))
            # -------------  Afișarea ariilor calculate--------------------
            # # for i, area in enumerate(areas):
            # #     print(f'Aria conturului {i + 1}: {area:.2f}')
            
            # # Vizualizarea graficului 3D și a planului de intersecție
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.plot_surface(x, y, z, cmap='gray', alpha=0.5)
            
            # # Adăugarea contururilor pe grafic
            # for contour in filtered_contours:
            #     x_contour = contour[:, 1]
            #     y_contour = contour[:, 0]
            #     z_contour = np.full_like(x_contour, z_intersect)
            #     ax.plot(x_contour, y_contour, z_contour, color='red')
            
            # # Personalizare grafic
            # ax.set_title('Grafic 3D din Imagine PNG cu Plan de Intersecție')
            # ax.set_xlabel('X axis')
            # ax.set_ylabel('Y axis')
            # ax.set_zlabel('Grayscale Value')
            
            # plt.show()
            # ---------   end afisare ---------------------------------------
#  -------------------------------------------------------
            diff_filename = f"diff_{filename}"
            diff_path = os.path.join(diff_folder, diff_filename)
            cv2.imwrite(diff_path, dilated_grayscale)
            image_array = np.array(dilated_grayscale)
            data.append({"Index": index, "Filename": filename, "Original": image, "Image_Array": image_array, "contours":shrunken_contours} )
            data_contours.append({"Index": index,"contours":contours_cnt, "Original": image,})
    
        previous_image = image
    df = pd.DataFrame(data)
    dc = pd.DataFrame(data_contours)
    return df,dc, image_mask


def compute_contour_stats(df):
    """
    Computes a dictionary where:
    - Key: Index from the DataFrame
    - Value: [total surface area of all contours, number of contours]
    
    :param df: Pandas DataFrame containing 'Index' and 'contours' columns.
    :return: Dictionary {Index: [total_surface, num_contours]}
    """
    contour_stats = {}

    for _, row in df.iterrows():
        index = row["Index"]
        contours = row["contours"]

        # Compute total area of all contours
        total_surface = sum(cv2.contourArea(contour.astype(np.float32)) for contour in contours) if contours else 0
        num_contours = len(contours)  # Number of contours

        # Store in dictionary as a list [total_surface, num_contours]
        contour_stats[index] = [total_surface, num_contours, total_surface / num_contours if num_contours > 0 else 0]

    return contour_stats

def draw_contours_matplotlib(image_array, contours):
    """
    Draws contours on the given image using Matplotlib.
    
    :param image_array: The grayscale or color image (NumPy array).
    :param contours: List of contours to draw.
    """
    # Convert image to grayscale if it's RGB
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = np.mean(image_array, axis=2)  # Convert to grayscale

    # Create figure
    plt.figure(figsize=(6, 6))
    plt.imshow(image_array, cmap='gray', origin='upper')

    # Draw each contour
    for contour in contours:
        contour = np.array(contour)  # Ensure it's a NumPy array
        if contour.shape[1] == 2:  # Ensure valid shape (N, 2)
            plt.plot(contour[:, 1], contour[:, 0], color='lime', linewidth=1)  # Swap X, Y

    # Display plot
    plt.title("Contours Overlaid on Image")
    plt.axis("off")
    plt.show()



def draw_contours(image_array, contours):
    """
    Draws contours on the given image.
    
    :param image_array: The grayscale or color image (NumPy array).
    :param contours: List of contours to draw.
    :return: Image with contours drawn.
    """
    # Convert grayscale image to BGR if needed
    if len(image_array.shape) == 2:  # Grayscale image
        image_with_contours = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    else:
        image_with_contours = image_array.copy()

    processed_contours = preprocess_contours(contours)
    if processed_contours:
        cv2.drawContours(image_with_contours, processed_contours, -1, (0, 255, 0), 1)
    else:
        print("No valid contours found to draw.")

    return image_with_contours

def contours_intersect(contour1, contour2):
    """
    Checks if two contours intersect using OpenCV's pointPolygonTest.

    :param contour1: First contour (NumPy array of shape (N, 1, 2)).
    :param contour2: Second contour (NumPy array of shape (M, 1, 2)).
    :return: True if contours intersect, False otherwise.
    """
    # Check if any point of contour1 is inside contour2
    for point in contour1:
        x, y = float(point[0]), float(point[1])
        # print(point)
        if cv2.pointPolygonTest(contour2, (x,y), False) >= 0:
            return True

    # Check if any point of contour2 is inside contour1
    for point in contour2:
        if cv2.pointPolygonTest(contour1, (x,y), False) >= 0:
            return True

    return False  # No intersection

def create_input_band (image_height, image_width):
    """
    Creates a rectangular contour with:
    - Full image width.
    - Height ranging from 90% to 75% of the total image height.

    :param image_height: Height of the image.
    :param image_width: Width of the image.
    :return: NumPy array representing the rectangle contour.
    """
    y_top = int(image_height * 0.75)  # 75% of height
    y_bottom = int(image_height * 0.90)  # 90% of height

    # Define the rectangle points (clockwise order)
    rectangle_contour = np.array([
        [0, y_top],  # Top-left corner
        [image_width - 1, y_top],  # Top-right corner
        [image_width - 1, y_bottom],  # Bottom-right corner
        [0, y_bottom]  # Bottom-left corner
    ], dtype=np.int32)

    return rectangle_contour

def check_in_proximity_by_centroid(contour1, contour2, factor=1.2):
    """
    Checks if two contours are in proximity based on the distance between their centroids.

    :param contour1: First contour (NumPy array of shape (N, 1, 2)).
    :param contour2: Second contour (NumPy array of shape (M, 1, 2)).
    :param factor: Multiplication factor (default 1.2) for vertical size check.
    :return: True if the contours are in proximity, False otherwise.
    """
   
    # Compute centroids of contours
    centroid1 = np.mean(contour1, axis=0)  # [x, y] center of contour1
    centroid2 = np.mean(contour2, axis=0)  # [x, y] center of contour2
    
    # Compute vertical distance between centroids
    vertical_distance = abs(centroid1[1] - centroid2[1])

    # Extract min and max Y-coordinates for each contour
    y_min1, y_max1 = np.min(contour1[:, 0]), np.max(contour1[:, 0])
    y_min2, y_max2 = np.min(contour2[:, 0]), np.max(contour2[:, 0])

    # Calculate the vertical sizes of the contours
    height1 = y_max1 - y_min1
    height2 = y_max2 - y_min2

    # Determine the maximum allowed vertical distance
    max_allowed_distance = factor * max(height1, height2)

    return vertical_distance <= max_allowed_distance

def check_in_proximity_by_centroid_for_insertion(contours, triplet, window):
    """
    The input represents the contour which appears in the inpt band
    :contours are all the triplets for one channel 
    """
    flag_index = 0
    flag = False
    persistence = 0
    frame_index = triplet[0]
    for cont in contours:
        
        if (cont[0] > frame_index) and (cont[0] < frame_index + window) :
            if check_in_proximity_by_centroid(cont[1],triplet[1]):
                print("in proximity")
                triplet = cont
                persistence=persistence+1   
                flag_index = cont[0]

    if persistence > 4:
        flag = True
       
    return flag, flag_index

def cluster_trajectories_from_lists(time_list, position_list, eps=3, min_samples=2):
    """
    Uses DBSCAN to cluster trajectories from a single list of time values and a single list of position values.

    :param time_list: List of time values (1D list).
    :param position_list: List of position values corresponding to the time values (1D list).
    :param eps: Maximum distance between points for them to be considered part of a cluster.
    :param min_samples: Minimum number of points required to form a cluster.
    :return: Dictionary mapping trajectory clusters to their time and position points.
    """
    # Ensure input lists are NumPy arrays
    time_array = np.array(time_list)
    position_array = np.array(position_list)

    # Combine time and position into a 2D array for DBSCAN
    trajectory_data = np.column_stack((time_array, position_array))

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(trajectory_data)

    # Organize trajectories based on cluster labels
    clustered_trajectories = {}
    for label, t, p in zip(cluster_labels, time_array, position_array):
        if label not in clustered_trajectories:
            clustered_trajectories[label] = {"time": [], "position": []}
        clustered_trajectories[label]["time"].append(t)
        clustered_trajectories[label]["position"].append(p)

    return clustered_trajectories

def plot_clustered_trajectories(clustered_trajectories, output_path):
    """
    Plots all clustered trajectories with different colors.

    :param clustered_trajectories: Dictionary mapping cluster labels to time and position lists.
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(clustered_trajectories)))

    for (label, data), color in zip(clustered_trajectories.items(), colors):
        plt.plot(data["time"], data["position"], marker="o", linestyle="-", label=f"Cluster {label}", color=color)

    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Clustered Trajectories")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def elongate_clustered_trajectories(clustered_trajectories, y_thr = 50, time = 7):
    
    clustered_trajectories_elongated = clustered_trajectories
    
    loop = True
    while loop == True:
        loop = False
        clustered_trajectories = clustered_trajectories_elongated
        for cluster, dict in clustered_trajectories.items():
            if cluster != -1 :
                tmp_key = cluster
                tmp_value = dict
                for cluster2, dict2 in clustered_trajectories.items():
                    if cluster2 != -1 and cluster2 != cluster:
                        a = False
                        b = False
                        for key, value in dict2.items():
                            if key == 'time':
                                if abs(value[0] - dict['time'][-1])<7:
                                    a = True
                            if key == 'position':
                                if abs(value[0] - dict['position'][-1])<50:
                                    b = True
                            if a and b :
                                tmp_value['time'][-1] = dict2['time'][0] 
                                tmp_value['time'].extend(dict2["time"][1:]) 
                                tmp_value['position'][-1] = dict2['position'][0] 
                                tmp_value["position"].extend(dict2["position"][1:]) 
                                del clustered_trajectories_elongated[cluster2]
                                loop = True
                                break
                    if loop:
                        break  # Break middle loop
                clustered_trajectories_elongated[tmp_key] = tmp_value # add the composed trajectory to elongated
                    
                if loop:
                    break  # Break outer loop
                
    return clustered_trajectories_elongated

def clean_clustered_trajectories(clustered_trajectories, size_thr = 4):
    
    for cluster in list(clustered_trajectories.keys()):  
        if cluster == -1 or len(clustered_trajectories[cluster]["time"]) <= size_thr:
            del clustered_trajectories[cluster]  # Safe deletion

    for cluster in list(clustered_trajectories.keys()):
        time_values = clustered_trajectories[cluster]["time"]
        position_values = clustered_trajectories[cluster]["position"]

        # Ensure time is sorted in ascending order
        cleaned_time = []
        cleaned_position = []
        
        previous_time = float('-inf')  # Initialize with a very small value

        for t, p in zip(time_values, position_values):
            if t > previous_time:  # Only keep values that maintain ascending order
                cleaned_time.append(t)
                cleaned_position.append(p)
                previous_time = t

        # Update the cluster with cleaned values
        clustered_trajectories[cluster]["time"] = cleaned_time
        clustered_trajectories[cluster]["position"] = cleaned_position

    return clustered_trajectories

def compute_speeds(clustered_trajectories):
    """
    Computes the instantaneous speed and average speed for each trajectory.

    :param clustered_trajectories: Dictionary of trajectory clusters with "time" and "position" lists.
    :return: Dictionary with added "instantaneous_speed" and "average_speed" for each trajectory.
    """
    for cluster in list(clustered_trajectories.keys()):
        time_values = np.array(clustered_trajectories[cluster]["time"])
        position_values = np.array(clustered_trajectories[cluster]["position"])

        if len(time_values) < 2:
            # If there's only one or no points, speed can't be computed
            clustered_trajectories[cluster]["instantaneous_speed"] = []
            clustered_trajectories[cluster]["average_speed"] = 0
            continue

        # Compute instantaneous speed: dx/dt
        time_differences = np.diff(time_values)
        position_differences = np.diff(position_values)
        instantaneous_speeds = position_differences / time_differences

        # Compute average speed: total displacement / total time
        total_displacement = position_values[-1] - position_values[0]
        total_time = time_values[-1] - time_values[0]
        average_speed = total_displacement / total_time if total_time != 0 else 0

        # Store results
        clustered_trajectories[cluster]["instantaneous_speed"] = instantaneous_speeds.tolist()
        clustered_trajectories[cluster]["average_speed"] = average_speed

    return clustered_trajectories


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    image_folder = sys.argv[1]
    if not os.path.isdir(image_folder):
        print(f"Error: {image_folder} is not a valid directory.")
        sys.exit(1)
    
    df,dc,image_mask = process_images(image_folder)

    print(df.head())
    num_rows = df.shape[0]
    print("Number of rows in the DataFrame:", num_rows)

    contour_statistics = compute_contour_stats(df)

    indices = np.array(list(contour_statistics.keys()))
    num_contours = np.array([contour_statistics[idx][1] for idx in indices])

    quartic_coeffs = np.polyfit(indices, num_contours, 4)  # Quartic polynomial fit
    quartic_poly = np.poly1d(quartic_coeffs)  # Create polynomial function  
    x_smooth = np.linspace(min(indices), max(indices), 300)
    y_quartic_smooth = quartic_poly(x_smooth)  # Quartic polynomial values


    residuals = num_contours - quartic_poly(indices)
    # Compute standard deviation and remove 10% extreme outliers
    std_dev = np.std(residuals)
    threshold = 1.5 * std_dev  # 90% confidence interval (removes 10% of extreme outliers)
    upper_bound = quartic_poly(x_smooth) + threshold
    lower_bound = quartic_poly(x_smooth) - threshold
    outlier_mask = np.abs(residuals) > threshold
    outlier_indices = indices[outlier_mask]
    print("Outlier Indices:", outlier_indices.tolist())


    df_filtered = df[~df["Index"].isin(outlier_indices)]
    dc_filtered = dc[~dc["Index"].isin(outlier_indices)]
    # output_path = os.path.join(image_folder, "plots", "output.pkl")
    # df.to_pickle(output_path)

    contour_groups_dict = group_contours_by_x_position(dc_filtered, tolerance=10)
    print("Numar de canale = ",len(contour_groups_dict))
    height, width = image_mask.shape  
    input_band = create_input_band(height, width)

    print(f"Height: {height}, Width: {width}")
    print("Rectangle Contour Points:\n", input_band)
    # ----------- Plot all the y and centers for a given channel -----------------
    for group_id, contours in contour_groups_dict.items():
        x_values = []
        y_values = []
        for index, cnt in enumerate(contours):
            print("cell ", index, " el = ", cnt[0],  cnt[2])
            try:
                y_value = cnt[2][0]  # Access element[2][1]
                x_values.append(cnt[0])
                y_values.append(y_value)
            except (IndexError, TypeError):
                print(f"Skipping element at index {index}: Invalid structure.")
        
        clustered_objects = cluster_trajectories_from_lists(x_values, y_values, eps=12, min_samples=2)
        print("\nCluster Assignments:", clustered_objects)
        # plot_clustered_trajectories(clustered_objects)

        elongated_trajectories = elongate_clustered_trajectories(clustered_objects)
        elongated_trajectories = clean_clustered_trajectories(elongated_trajectories)

        updated_trajectories = compute_speeds(elongated_trajectories)
        df_speeds = pd.DataFrame.from_dict(updated_trajectories, orient="index")

        csv_file_path = os.path.join(image_folder, "plots", f"clustered_trajectories_with_speed_{group_id}.csv")
        df_speeds.to_csv(csv_file_path, index=True)
        # tools.display_dataframe_to_user(name="Clustered Trajectories with Speed", dataframe=df_speeds)
        output_path = os.path.join(image_folder, "plots", f"Cells_Trajectories_channel_{group_id}.png")
        plot_clustered_trajectories(elongated_trajectories, output_path)
        

        plt.figure(figsize=(8, 5))
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label="Channel cells")
        plt.xlabel("Time Index")
        plt.ylabel("Channe Y position")
        plt.title(f"Plot of Index vs Y for channel '{group_id}'")
        plt.legend()
        plt.grid(True)
        output_path = os.path.join(image_folder, "plots", f"Cells_channel_{group_id}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    # ----------------------------------------------------------------------------
        

    
        # --------------------  ploting contours on the same vertical ------------------
        # if group_id == 6: 
        #     print(f"Group {group_id} (Size: {len(contours)}):")
            
        #     for contour in contours:
        #         print(contour)
        #         # print(f"  - Index {np.array(contour[1])}, Contour Shape: {contour[2]}")
        #         if len(image_mask.shape) == 3 and image_mask.shape[2] == 3:
        #             image_mask = np.mean(image_mask, axis=2)
        #         plt.figure(figsize=(6, 6))
        #         plt.imshow(image_mask, cmap='gray', origin='upper')
        #         cnt = np.array(contour[1])
        #         plt.plot(cnt[:, 1], cnt[:, 0],color='lime', linewidth=1)

        #            # Display plot
        #         plt.title("Contours Overlaid on Image")
        #         plt.axis("off")
        #         plt.show()
  
    # for _, row in dc_filtered.iterrows():
    #     if row["contours"]:  # Ensure there are contours to draw
    #         print(row["contours"])
    #         draw_contours_matplotlib(row["Original"], row["contours"])

    num_rows = df_filtered.shape[0]
    print("Number of rows in the filtered DataFrame:", num_rows)

    # for _, row in df_filtered.iterrows():
    #     if row["contours"]:  # Ensure there are contours to draw
    #         print(row["contours"])
    #         draw_contours_matplotlib(row["Original"], row["contours"])    
            # image_with_contours = draw_contours(row["Original"], row["contours"])


    # Plot the original data points
    plt.figure(figsize=(8, 5))
    plt.scatter(indices, num_contours, color='red', label="Original Data (Samples)", zorder=3)
    plt.plot(x_smooth, y_quartic_smooth, color='blue', label="Quartic Fit", zorder=2)

    # Draw the outlier elimination surface (shaded region)
    plt.fill_between(x_smooth, lower_bound, upper_bound, color='blue', alpha=0.2, label="Outlier Removal Region")

    # Labels and title
    plt.xlabel("Index")
    plt.ylabel("Number of Contours")
    plt.title("Quartic Polynomial Fit with Outlier Removal Region") 
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.join(image_folder, "plots"), exist_ok=True)
    output_path = os.path.join(image_folder, "plots", "outlier_detection_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()
   

    # Generate smooth curve using spline interpolation
    x_smooth = np.linspace(min(indices), max(indices), 300)  # Smooth x values
    spline = make_interp_spline(indices, num_contours, k=1)  # Cubic spline interpolation
    y_smooth = spline(x_smooth)

    # Plot the original data points
    plt.figure(figsize=(8, 5))
    plt.scatter(indices, num_contours, color='red', label="Original Data (Samples)", zorder=3)
    plt.plot(x_smooth, y_smooth, color='blue', label="Spline Interpolation", zorder=2)

    # Labels and title
    plt.xlabel("Index")
    plt.ylabel("Number of Contours")
    plt.title("Number of Contours vs. Index (With Spline Interpolation)")
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(image_folder, "plots", "contour_vs_index.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()


    
    indices = np.array(list(contour_statistics.keys()))
    num_contours = np.array([contour_statistics[idx][0] for idx in indices])
    # Generate smooth curve using spline interpolation
    x_smooth = np.linspace(min(indices), max(indices), 300)  # Smooth x values
    spline = make_interp_spline(indices, num_contours, k=1)  # Cubic spline interpolation
    y_smooth = spline(x_smooth)

    # Plot the original data points
    plt.figure(figsize=(8, 5))
    plt.scatter(indices, num_contours, color='red', label="Original Data (Samples)", zorder=3)
    plt.plot(x_smooth, y_smooth, color='green', label="Spline Interpolation", zorder=2)

    # Labels and title
    plt.xlabel("Index")
    plt.ylabel("Cumulative cells size")
    plt.title("Cumulative cells size vs. Index (With Spline Interpolation)")
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(image_folder, "plots", "cumulCellSize_vs_index.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()

    indices = np.array(list(contour_statistics.keys()))
    num_contours = np.array([contour_statistics[idx][2] for idx in indices])

    # Generate smooth curve using spline interpolation
    x_smooth = np.linspace(min(indices), max(indices), 300)  # Smooth x values
    spline = make_interp_spline(indices, num_contours, k=1)  # Cubic spline interpolation
    y_smooth = spline(x_smooth)

    # Plot the original data points
    plt.figure(figsize=(8, 5))
    plt.scatter(indices, num_contours, color='red', label="Original Data (Samples)", zorder=3)
    plt.plot(x_smooth, y_smooth, color='yellow', label="Spline Interpolation", zorder=2)

    # Labels and title
    plt.xlabel("Index")
    plt.ylabel("Average Cell Size")
    plt.title("Average Cell Size vs. Index (With Spline Interpolation)")
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(image_folder, "plots", "avgCellSize_vs_index.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()
    # Print results
    print(contour_statistics)

if __name__ == "__main__":
    main()


# ------------------------------ to do ----------------------------
#   - on the data frame, eliminate images that have contours with more than 60 percent different then the next and previous frame 
# are different 
#   - remove concave parts
#   - unify contours that are close to each other on y axis
