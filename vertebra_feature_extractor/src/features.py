
'''
Generate height maps from vertebral body meshes.
Meshes must be either all in the same folder or in a folder structure such as:
Parent folder: {
    case folder 1: {
        meshes: {
            mesh file (.stl)
            ...
        }
    }
    case folder 2: {
        meshes: {
            mesh file (.stl)
            ...
        }
    }
    ...
}


Saves relevant parameters extracted from the height maps in a csv file.
'''

import pyvista as pv
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R
import os
import cv2
import pandas as pd
from scipy.ndimage import center_of_mass
import logging
from pathlib import Path
import pandas as pd
from rich.progress import track
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from rich.logging import RichHandler
from rich.progress import track

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger(__name__)

def find_plane_of_symmetry(mesh, idx=1):
    # Step 1: Center the mesh
    vertices = np.array(mesh.points)
    centroid = vertices.mean(axis=0)
    centered_vertices = vertices - centroid

    # Step 2: Perform PCA
    pca = PCA(n_components=3)
    pca.fit(centered_vertices)
    
    # The third principal component is perpendicular to the plane spanned by the first two
    normal_vector = pca.components_[idx]

    # Step 3: Define the plane passing through the centroid
    plane_point = centroid  # Use the centroid as a point on the plane

    return plane_point, normal_vector

def find_optimal_process_CoM(process_mesh, vb_mesh, inferior_vector):
    vb_CoM = vb_mesh.center
    process_CoM = process_mesh.center

    dist = 100

    while dist > 1:
        coronal_normal = np.array(vb_CoM) - np.array(process_CoM)
        saggital_normal = np.cross(coronal_normal, inferior_vector) 

        line = pv.Line(process_CoM, vb_CoM)
        points, intersection_ind = vb_mesh.ray_trace(process_CoM, vb_CoM)

        if points.size > 0:
            # Take the first intersection point
            first_intersection_point = points[0]
            
        else:
            log.info("No intersection found between the line and vb_mesh.")
            return process_mesh

        # delete the points in the process mesh that are further than the plane defined by the intersection point and the coronal normal
        coronal_normal = coronal_normal / np.linalg.norm(coronal_normal)
        distances = np.dot(process_mesh.points - first_intersection_point, coronal_normal)

        # Filter out points in process_mesh that are further than the intersection plane
        filtered_points = process_mesh.points[distances <= 0]

        # check the distance between the previous CoM and the new CoM
        dist = np.linalg.norm(np.array(process_CoM) - np.array(filtered_points.mean(axis=0)))
        process_CoM = filtered_points.mean(axis=0)

    filtered_mesh = pv.PolyData(filtered_points)
        
    return filtered_mesh

def reflect_mesh(mesh, plane_origin, plane_normal):
    mirror_matrix = np.eye(4)
    mirror_matrix[:3, :3] -= 2 * np.outer(plane_normal, plane_normal)

    # translate the mesh to the symmetry plane origin, apply mirror and translate back
    translation_to_origin = -plane_origin
    translation_back = plane_origin
    mirroed_points = (mesh.points + translation_to_origin) @ mirror_matrix[:3, :3].T + translation_back

    original_com = np.mean(mesh.points, axis=0)  # Center of mass of the original mesh
    mirrored_com = np.mean(mirroed_points, axis=0)  # Center of mass of the mirrored mesh
    com_correction = original_com - mirrored_com  # Offset to align centers of mass
    mirroed_points += com_correction
    
    # create mirrored mesh
    mirrored_mesh = pv.PolyData(mirroed_points)

    # Find the closest point in the original mesh for each mirrored point
    kd_tree = cKDTree(mesh.points)  # Build a k-d tree for fast nearest-neighbor search
    distances, indices = kd_tree.query(mirroed_points)

    return mirrored_mesh, distances

def compute_heights(rotated_mesh, grid_res):
    # Compute grid resolution based on the mesh dimensions
    x_min, x_max = np.min(rotated_mesh[:, 0]), np.max(rotated_mesh[:, 0])
    y_min, y_max = np.min(rotated_mesh[:, 1]), np.max(rotated_mesh[:, 1])
    z_min, z_max = np.min(rotated_mesh[:, 2]), np.max(rotated_mesh[:, 2])

    grid_size_post = (int(grid_res * (x_max - x_min)), int(grid_res * (y_max - y_min)))
    grid_size_inf = (int(grid_res * (x_max - x_min)), int(grid_res * (z_max - z_min)))
    grid_size_lat = (int(grid_res * (z_max - z_min)), int(grid_res * (y_max - y_min)))

    X_post, Y_post = np.meshgrid(np.linspace(x_min, x_max, grid_size_post[0]), np.linspace(y_min, y_max, grid_size_post[1]))
    Z_lat, Y_lat = np.meshgrid(np.linspace(z_min, z_max, grid_size_lat[0]), np.linspace(y_min, y_max, grid_size_lat[1]))
    Z_inf, X_inf = np.meshgrid(np.linspace(z_min, z_max, grid_size_inf[0]), np.linspace(x_min, x_max, grid_size_inf[1]))
    
    # Create an empty array for heights
    heights = np.zeros((grid_size_post[1], grid_size_post[0]))
    bin_mask = np.zeros((grid_size_post[1], grid_size_post[0]))
    inferior_distances = np.zeros((grid_size_post[1], grid_size_post[0]))
    lat_distances = np.zeros((grid_size_lat[1], grid_size_lat[0]))
    posterior_distances = np.zeros((grid_size_inf[1], grid_size_inf[0]))

    for i in range(grid_size_post[1]-1):
        for j in range(grid_size_post[0]-1):
            # Find all the points contained in the grid cell
            mask = (rotated_mesh[:, 0] >= X_post[i, j]) & (rotated_mesh[:, 0] <= X_post[i + 1, j + 1]) & \
                (rotated_mesh[:, 1] >= Y_post[i, j]) & (rotated_mesh[:, 1] <= Y_post[i + 1, j + 1])
            if np.any(mask):
                z_values = rotated_mesh[mask, 2]
                height = np.max(z_values) - np.min(z_values)
                heights[i, j] = height
                bin_mask[i, j] = int(z_values.shape[0] > 0)
                inferior_distances[i, j] = np.max(z_values)

    for i in range(grid_size_lat[1]-1):
        for j in range(grid_size_lat[0]-1):
            # Find all the points contained in the grid cell
            mask = (rotated_mesh[:, 2] >= Z_lat[i, j]) & (rotated_mesh[:, 2] <= Z_lat[i + 1, j + 1]) & \
                (rotated_mesh[:, 1] >= Y_lat[i, j]) & (rotated_mesh[:, 1] <= Y_lat[i + 1, j + 1])
            if np.any(mask):
                x_values = rotated_mesh[mask, 0]
                distance = np.min(x_values)
                lat_distances[i, j] = np.abs(distance)
    
    for i in range(grid_size_inf[1] - 1):
        for j in range(grid_size_inf[0] - 1):
            # Find all the points contained in the grid cell
            mask = (rotated_mesh[:, 2] >= Z_inf[i, j]) & (rotated_mesh[:, 2] <= Z_inf[i + 1, j + 1]) & \
                (rotated_mesh[:, 0] >= X_inf[i, j]) & (rotated_mesh[:, 0] <= X_inf[i + 1, j + 1])
            if np.any(mask):
                x_values = rotated_mesh[mask, 1]
                distance = np.min(x_values)
                posterior_distances[i, j] = np.abs(distance)
    

    return heights, bin_mask, inferior_distances, posterior_distances, lat_distances

def get_main_axis(distance_projections, mesh_path):
    # Do PCA on the posterior_distances to find the main axis
    contours, _ = cv2.findContours(distance_projections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if there are more than one contour with area bigger than 10% of the image, then continue
    if len(contours) > 1:
        areas = [cv2.contourArea(contour) for contour in contours]
        big_areas = [area for area in areas if area >= 0.1*distance_projections.shape[0]*distance_projections.shape[1]]
        if len(big_areas) > 1:
            # with open('/home/blanca/Documents/height_maps/WHOLE_PIPELINE/tab_data/Spine1K/colon/multiple_contours.txt', 'a') as f:
            #     f.write(mesh_path + '\n')
            return None, None
        else:
            max_area_index = np.argmax(areas)
            contour = contours[max_area_index]
    elif len(contours) == 1:
        contour = contours[0]
    else:
        # write in a file
        # with open('/home/blanca/Documents/height_maps/WHOLE_PIPELINE/tab_data/Spine1K/colon/no_contours.txt', 'a') as f:
        #     f.write(mesh_path + '\n')
        return None, None
    
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # The vectors of the box edges can be taken as axes
    axis1 = box[1] - box[0]
    axis2 = box[2] - box[1]
    # Calculate the center of the rectangle to draw the axes
    center = np.mean(box, axis=0)
    
    # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # ax[0].imshow(distance_projections, cmap='viridis')
    # ax[0].axis('off')
    # ax[1].imshow(distance_projections, cmap='viridis')
    # display the main axis (axis1 and axis2)
    # Draw the principal axes
    # scale_factor = 0.5  # Adjust this to control the length of the displayed axes
    # ax[1].arrow(center[0], center[1], scale_factor * axis1[0], scale_factor * axis1[1], color='red', head_width=3, head_length=6)
    # ax[1].arrow(center[0], center[1], scale_factor * axis2[0], scale_factor * axis2[1], color='blue', head_width=3, head_length=6)
    # ax[1].scatter(center[0], center[1], color='yellow')  # Mark the center point
    # ax[1].axis('off')
    # plt.show()
    # Normalize the principal axis vector
    norm_axis2 = axis2 / np.linalg.norm(axis2)
    norm_axis1 = axis1 / np.linalg.norm(axis1)

    # Define the image x-axis (1, 0) and y-axis (0, 1) vectors
    image_x_axis = np.array([1, 0])
    image_y_axis = np.array([0, 1])

    # Compute the angle with the x-axis using the dot product
    dot_product_y, dot_product_x = np.dot(norm_axis2, image_y_axis), np.dot(norm_axis1, image_x_axis)
    angle_with_y, angle_with_x = np.arccos(dot_product_y), np.arccos(dot_product_x)  # Angle in radians
    angle_with_y_degrees, angle_with_x_degrees = np.degrees(angle_with_y), np.degrees(angle_with_x)  # Convert to degrees

    return angle_with_x_degrees, angle_with_y_degrees

def create_heigth_map(mesh_path):
    # Read the mesh
    mesh = pv.read(mesh_path)

    # Triangulate the mesh if needed
    if not mesh.is_all_triangles:
        mesh.triangulate(inplace=True)
    
    # Compute normals
    mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    normals = mesh['Normals']

    # Number of main surfaces to find
    num_surfaces = 6  # Adjust based on your mesh

    # Fit KMeans
    kmeans = KMeans(n_clusters=num_surfaces, random_state=42)
    kmeans.fit(normals)
    labels = kmeans.labels_

    # Add cluster labels to the mesh
    mesh['Labels'] = labels

    # mesh.save('/home/blanca/Documents/spine_project/mesh_parameters/all_meshes_body/verse020_L7_with_labels.vtk')

    # Compute the main normal vector for each surface
    main_normals = []
    for i in range(num_surfaces):
        cluster_normals = normals[labels == i]
        main_normal = np.mean(cluster_normals, axis=0)
        main_normal /= np.linalg.norm(main_normal)  # Normalize
        main_normals.append(main_normal)

    # Inferior plane normal vector (assumed to be along the z-axis)
    inferior_normal = np.array([0, 0, -1])

    # Find surfaces that are approximately parallel to the inferior plane
    parallel_surfaces = []
    # get the dot product of the main normals with the inferior normal
    all_dot_products = [np.abs(np.dot(normal, inferior_normal)) for normal in main_normals]
    # keep the two surfaces with the highest dot product
    parallel_surfaces = np.argsort(all_dot_products)[-2:]

    # Extract points from the parallel surfaces and compute distances
    parallel_points = []
    for i in parallel_surfaces:
        surface_points = mesh.points[labels == i]
        parallel_points.append(surface_points)

    # Find the surface closer to the inferior plane
    min_distance = float('inf')
    closest_surface_index = None

    for i, points in enumerate(parallel_points):
        # Compute the distance of each point to the inferior plane
        distances = np.abs(points[:, 2])  # Since the inferior plane normal is along z-axis
        avg_distance = np.mean(distances)
        
        if avg_distance < min_distance:
            min_distance = avg_distance
            closest_surface_index = parallel_surfaces[i]

    # Save the closest parallel surface as a separate file
    if closest_surface_index is not None:

        # Calculate the normal of the closest surface
        closest_normal = main_normals[closest_surface_index]
        
        # Calculate the rotation needed to align the closest surface normal to the inferior plane normal
        projected_normal = np.array([0, closest_normal[1], closest_normal[2]])
        projected_normal /= np.linalg.norm(projected_normal)  # Normalize

        # Calculate the rotation needed to align the projected normal to the inferior plane normal
        rotation_vector = np.cross(projected_normal, inferior_normal)
        rotation_angle = np.arccos(np.dot(projected_normal, inferior_normal))
        # if the angle is between 0 and 1, add 180 degrees to the angle
        if 0 < rotation_angle < 1:
        #    with open('/home/blanca/Documents/height_maps/WHOLE_PIPELINE/tab_data/Spine1K/colon/flip.txt', 'a') as f:
        #        f.write(mesh_path + '\n')
            rotation_angle += np.pi
        
        if np.linalg.norm(rotation_vector) != 0:  # Ensure rotation vector is valid
            rotation = R.from_rotvec(rotation_angle * rotation_vector / np.linalg.norm(rotation_vector))
            rotation_matrix = rotation.as_matrix()
            
            # Apply the rotation to the mesh points
            rotated_mesh_points = mesh.points @ rotation_matrix.T
            
            # Translate the mesh to make sure it aligns properly with the inferior plane
            min_z = np.min(rotated_mesh_points[:, 2])
            translation_vector = np.array([0, 0, -min_z])
            translated_mesh_points = rotated_mesh_points + translation_vector
            translated_mesh = pv.PolyData(translated_mesh_points, mesh.faces)
            # save the rotated mesh 
            # translated_mesh.save('/home/blanca/Documents/height_maps/WHOLE_PIPELINE/align_2.vtk')

            # Compute heights
            heights_low, binary_mask_low, inferior_distances, posterior_distances, lat_distances = compute_heights(translated_mesh_points, 0.5)
            heights_high, binary_mask_high, _, _, _ = compute_heights(translated_mesh_points, 1)

            posterior_angle_x, posterior_angle_y = get_main_axis((posterior_distances>0).astype(np.uint8), mesh_path)
            inferior_angle_x, inferior_angle_y = get_main_axis((inferior_distances>0).astype(np.uint8), mesh_path)
            lateral_angle_x, lateral_angle_y = get_main_axis((lat_distances>0).astype(np.uint8), mesh_path)
            
            # reshape the binary_mask_low to have the same shape as the heights_high
            binary_mask_low = cv2.resize(binary_mask_low, (heights_high.shape[1], heights_high.shape[0]), interpolation=cv2.INTER_NEAREST)
            heights_low = cv2.resize(heights_low, (heights_high.shape[1], heights_high.shape[0]), interpolation=cv2.INTER_NEAREST)
            binary_mask_high = binary_mask_high*binary_mask_low
            # find the voxels in binary_mask_high that are zero and have value = 1 in binary_mask_low
            object_pixels = np.argwhere((binary_mask_high == 0) & (binary_mask_low == 1))
            # loop over the pixels in the object
            for i, j in object_pixels:
                roi = binary_mask_high[max(0, i-2):min(heights_high.shape[0], i+2), max(0, j-2):min(heights_high.shape[1], j+2)]
                if np.sum(roi) > 10:
                    binary_mask_high[i, j] = 1

            # loop over the pixels in the object
            vert_black_pixels = np.argwhere((binary_mask_high == 1) & (heights_high < 10) & (heights_low > 5))
            # assign the value of heights low to heights high in those cases
            for i, j in vert_black_pixels:
                heights_high[i, j] = heights_low[i, j]

            bounds = translated_mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)

            # Compute the dimensions
            width = bounds[1] - bounds[0]
            height = bounds[3] - bounds[2]
            depth = bounds[5] - bounds[4]

            return heights_high, binary_mask_high, [posterior_angle_x, posterior_angle_y], [inferior_angle_x, inferior_angle_y], [lateral_angle_x, lateral_angle_y], [width, height, depth]
        
    else:
        print(f"No parallel surface found for {mesh_path}")
        return None, None, None, None, None, None

def create_new_sections(heights_high):
    # keep only the values in the binary mask where the value is bigger than the third quartile
    third_quartile = np.percentile(heights_high, 30)
    heights_map_vert = np.where(heights_high > third_quartile, 1, 0).astype(np.uint8)
    # create a circle in the center of mass of the object
    # Calculate the center of mass
    center = center_of_mass(heights_map_vert)
    # Convert the center of mass to integer coordinates
    center = tuple(map(int, center))
    # Define the radius of the ellipse (based on the height and width of the object across the x and y axis)
    # x_length is the sum of pixels across the line that passes through the center of mass and is parallel to the x axis
    # y_length is the sum of pixels across the line that passes through the center of mass and is parallel to the y axis
    x_length = np.sum(heights_map_vert[center[0], :])
    y_length = np.sum(heights_map_vert[:, center[1]])
    radius_x = int(x_length*0.25)
    radius_y = int(y_length*0.25)
    # Draw the elipse on the binary mask
    elipse_mask = np.zeros_like(heights_map_vert, dtype=np.uint8)
    cv2.ellipse(elipse_mask, (center[1], center[0]), (radius_x, radius_y), 0, 0, 360, 255, thickness=-1)

    # generate another elipse that will take as radius_x the minimum between the upper and lower x radius (given by the center of mass and the limit of the heights_map_vert)
    x_upper_length = np.sum(heights_map_vert[center[0], center[1]:])
    x_lower_length = np.sum(heights_map_vert[center[0], :center[1]])
    radius_x_2 = min(x_upper_length, x_lower_length)
    y_right_length = np.sum(heights_map_vert[center[0]:, center[1]])
    y_left_length = np.sum(heights_map_vert[:center[0], center[1]])
    radius_y_2 = min(y_left_length, y_right_length)
    # Draw the elipse on the binary mask
    elipse_mask_2 = np.zeros_like(heights_map_vert, dtype=np.uint8)
    cv2.ellipse(elipse_mask_2, (center[1], center[0]), (radius_x_2, radius_y_2), 0, 0, 360, 255, thickness=-1)


    ## GENERATE SECTIONS
    # Section 1 (top left)
    s1 = np.zeros_like(heights_high)
    s1[:center[0], :center[1]] = heights_high[:center[0], :center[1]]
    # s1 is where elipse mask is 0 and elipse mask 2 is 255
    s1 = np.where(elipse_mask == 0, s1, 0)
    s1 = np.where(elipse_mask_2 == 255, s1, 0)
    # Section 2 (top right)
    s3 = np.zeros_like(heights_high)
    s3[:center[0], center[1]:] = heights_high[:center[0], center[1]:]
    # s3 is where elipse mask is 0 and elipse mask 2 is 255
    s3 = np.where(elipse_mask == 0, s3, 0)
    s3 = np.where(elipse_mask_2 == 255, s3, 0)
    # Section 4 (bottom left)
    s4 = np.zeros_like(heights_high)
    s4[center[0]:, :center[1]] = heights_high[center[0]:, :center[1]]
    # s4 is where elipse mask is 0 and elipse mask 2 is 255
    s4 = np.where(elipse_mask == 0, s4, 0)
    s4 = np.where(elipse_mask_2 == 255, s4, 0)
    # Section 6 (bottom right)
    s6 = np.zeros_like(heights_high)
    s6[center[0]:, center[1]:] = heights_high[center[0]:, center[1]:]
    # s6 is where elipse mask is 0 and elipse mask 2 is 255
    s6 = np.where(elipse_mask == 0, s6, 0)
    s6 = np.where(elipse_mask_2 == 255, s6, 0)

    ## center sections
    s0_a = np.zeros_like(heights_high)
    s0_a[:center[0], :] = heights_high[:center[0], :]
    # s1 is where elipse mask is 0 and elipse mask 2 is 255
    s0_a = np.where(elipse_mask_2 == 255, s0_a, 0)
    s0_a = np.where(elipse_mask == 255, s0_a, 0)
    s0_p = np.zeros_like(heights_high)
    s0_p[center[0]:, :] = heights_high[center[0]:, :]
    # s3 is where elipse mask is 0 and elipse mask 2 is 255
    s0_p = np.where(elipse_mask_2 == 255, s0_p, 0)
    s0_p = np.where(elipse_mask == 255, s0_p, 0)
    s0_l = np.zeros_like(heights_high)
    s0_l[:, :center[1]] = heights_high[:, :center[1]]
    # s4 is where elipse mask is 0 and elipse mask 2 is 255
    s0_l = np.where(elipse_mask_2 == 255, s0_l, 0)
    s0_l = np.where(elipse_mask == 255, s0_l, 0)
    s0_r = np.zeros_like(heights_high)
    s0_r[:, center[1]:] = heights_high[:, center[1]:]
    # s6 is where elipse mask is 0 and elipse mask 2 is 255
    s0_r = np.where(elipse_mask_2 == 255, s0_r, 0)
    s0_r = np.where(elipse_mask == 255, s0_r, 0)

    elipse_values = heights_high[elipse_mask == 255]

    elipse_mask_2 = heights_high * elipse_mask_2
    elipse_mask = heights_high * elipse_mask

    return heights_map_vert, elipse_values, s1, s3, s4, s6, s0_a, s0_p, s0_l, s0_r, radius_x_2, radius_y_2, elipse_mask_2, elipse_mask

def extract_features(config: DictConfig):
    '''
    Extract features from the meshes in the parent_dir and save them in the output_dir.
    
    Args:
        parent_dir (Path): Parent directory containing the meshes.
        output_dir (Path): Directory to save the extracted features.

    '''
    parent_dir = Path(config.parent_dir)
    output_dir = Path(config.output_dir)
    input_csv = Path(config.input_csv)
    output_csv = Path(config.output_csv)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    df = pd.DataFrame(columns=['mesh_path', 'average_distance', 'max_distance', 'rms_distance'])
    
    cases = [case for case in parent_dir.iterdir() if case.is_dir()]
    prev_df_path = input_csv
    # idx = cases.index(Path('/data2/blanca/NMDID_hr/case-100155'))
    # cases = cases[:idx]
    if prev_df_path.exists():
        df = pd.read_csv(prev_df_path)
    else:
        df = pd.DataFrame(columns=['mesh_path'])
    if all(['stl' in str(case) for case in cases]):
        cases = ['']
    for case in track(cases, description=f'Extracting features from {len(cases)} cases'):
        if case != '':
            mesh_dir = case / 'meshes'
            if not mesh_dir.exists():
                log.error(f"Mesh directory not found for {mesh_dir}")
                continue
            meshes = [mesh for mesh in mesh_dir.iterdir() if mesh.suffix == '.stl' and 'body' in mesh.stem]
        else:
            meshes = [mesh for mesh in parent_dir.iterdir() if mesh.suffix == '.stl' and 'body' in mesh.stem]
        for mesh in meshes:
            if str(mesh) in df['mesh_path'].values:
                log.info(f"Skipping {mesh}")
                continue

            ##### SYMMETRY METRICS #####

            # check that the inferior vector, process and vertebra exist
            vert_mesh_path = mesh.parent / f'{mesh.stem.replace("body_vertebrae", "vertebrae")}.stl'
            vb_mesh_path = mesh
            if not vert_mesh_path.exists():
                log.error(f"Vertebra mesh not found for {mesh}")
                continue
            if not vb_mesh_path.exists():
                log.error(f"Vertebral body mesh not found for {mesh}")

            vert_mesh = pv.read(vert_mesh_path)
            vb_mesh = pv.read(vb_mesh_path)

            plane_point, symmetry_normal = find_plane_of_symmetry(vert_mesh)
            symmetry_plane = pv.Plane(center=plane_point, direction=symmetry_normal, i_size=100, j_size=100)

            # create a plane based on the mesh coordinates [0,0,1]
            frame_normal = np.array([1, 0, 0])
            frame_plane = pv.Plane(center=vb_mesh.center, direction=frame_normal, i_size=100, j_size=100)

            angle = np.arccos(np.dot(symmetry_normal, frame_normal) / (np.linalg.norm(symmetry_normal) * np.linalg.norm(frame_normal)))

            log.debug(f"angle between symmetry normal and frame normal: {np.degrees(angle)}")
            if np.degrees(angle) > 10 and np.degrees(angle) < 160:
                log.debug("Choosing the frame plane")
                chosen_plane = frame_plane
                chosen_normal = frame_normal
            else:
                chosen_plane = symmetry_plane
                chosen_normal = symmetry_normal

            mirroed_mesh, distances = reflect_mesh(vert_mesh, np.array(vb_mesh.center), chosen_normal)

            # Calculate symmetry measures
            average_distance = np.mean(distances)
            max_distance = np.max(distances)
            rms_distance = np.sqrt(np.mean(np.array(distances) ** 2))
            log.debug(f"average_distance: {average_distance}, max_distance: {max_distance}, rms_distance: {rms_distance}")
            
            try:
                mirroed_mesh, distances = reflect_mesh(vb_mesh, np.array(vb_mesh.center), chosen_normal)
                # Calculate symmetry measures
                average_distance_vb = np.mean(distances)
                max_distance_vb = np.max(distances)
                rms_distance_vb = np.sqrt(np.mean(np.array(distances) ** 2))
                log.debug(f"average_distance: {average_distance_vb}, max_distance: {max_distance_vb}, rms_distance: {rms_distance_vb}")
            except Exception as e:
                log.error(f"Error processing {mesh}: {e}")
                # if the mesh is not empty, display it
                if vb_mesh.n_points > 0:
                    p = pv.Plotter()
                    p.add_mesh(vb_mesh, color='red')
                    p.show()
                continue
            

            ### HEIGHT METRICS ###

            try:
                heights, binary_mask_high, posterior_angles, inferior_angles, lateral_angles, res = create_heigth_map(vb_mesh_path)
            except Exception as e:
                log.error(f"Error processing {mesh}: {e}")
                # add an empty row to the dataframe
                new_row = {'mesh_path': str(mesh)}
                df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
                continue

            if heights is None:
                log.error(f"Could not create height map for {mesh}")
                continue

            # normalize the heights
            heights = heights / np.max(heights)

            _, s0, s1, s3, s4, s6, s0_a, s0_p, s0_l, s0_r, x_rad_elipse, y_rad_elipse, big_elipse, small_elipse = create_new_sections(heights)
            
            bin_heights = np.where(heights > 0, 1, 0)
            dice = quantify_symmetry(bin_heights)
            all_stdv = np.std(heights[heights > 0])

            '''
            # Display the height map and the sections
            fig, ax = plt.subplots(2, 4, figsize=(15, 10))
            fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing between subplots

            # Row 0
            ax[0, 0].imshow(heights, cmap='viridis')
            ax[0, 0].axis('off')
            ax[0, 0].set_title('Height Map')

            ax[0, 1].imshow(small_elipse, cmap='viridis')
            ax[0, 1].axis('off')
            ax[0, 1].set_title('Small Ellipse')

            ax[0, 2].imshow(s1, cmap='viridis')
            ax[0, 2].axis('off')
            ax[0, 2].set_title('S1')

            ax[0, 3].imshow(s3, cmap='viridis')
            ax[0, 3].axis('off')
            ax[0, 3].set_title('S3')

            # Row 1
            ax[1, 0].imshow(s4, cmap='viridis')
            ax[1, 0].axis('off')
            ax[1, 0].set_title('S4')

            ax[1, 1].imshow(s6, cmap='viridis')
            ax[1, 1].axis('off')
            ax[1, 1].set_title('S6')

            ax[1, 2].imshow(big_elipse, cmap='viridis')
            ax[1, 2].axis('off')
            ax[1, 2].set_title('Big Ellipse')

            # Empty subplot for the last grid cell
            ax[1, 3].axis('off')  # Turn off the axis for the empty subplot

            # Show the figure
            plt.show()
            '''


            # compute the average value for each section including all the values of the height map where the height is bigger than 5 and the section is 255
            s0_avg = np.mean(s0[s0 > 0]) # set 0 to 5 if the heights are not normalized
            s1_avg = np.mean(s1[s1 > 0])
            s3_avg = np.mean(s3[s3 > 0])
            s4_avg = np.mean(s4[s4 > 0])
            s6_avg = np.mean(s6[s6 > 0])
            s0_a_avg = np.mean(s0_a[s0_a > 0])
            s0_p_avg = np.mean(s0_p[s0_p > 0])
            s0_l_avg = np.mean(s0_l[s0_l > 0])
            s0_r_avg = np.mean(s0_r[s0_r > 0])
            ant = s1+s3
            post = s4+s6
            ant_avg = np.mean(ant[ant > 0])
            post_avg = np.mean(post[post > 0])
            center_std = np.std(s0[s0 > 0])
            ant_stdv = np.std(ant[ant > 0])
            post_stdv = np.std(post[post > 0])

            new_row = {'mesh_path': str(mesh), 's0_avg': s0_avg, 's1_avg': s1_avg, 's3_avg': s3_avg, 's4_avg': s4_avg, 
                    's6_avg': s6_avg, 's0_a_avg': s0_a_avg, 's0_p_avg': s0_p_avg, 's0_l_avg': s0_l_avg, 's0_r_avg': s0_r_avg, 
                    'ant_avg': ant_avg, 'post_avg': post_avg, 'center_std': center_std, 'ant_stdv': ant_stdv, 'post_stdv': post_stdv,
                        'W':2*y_rad_elipse, 'posterior_angle_x': posterior_angles[0], 'posterior_angle_y': posterior_angles[1], 
                        'inferior_angle_x': inferior_angles[0], 'inferior_angle_y': inferior_angles[1], 'lateral_angle_x': lateral_angles[0], 
                        'lateral_angle_y': lateral_angles[1], 'res_x': res[0], 'res_y': res[1], 'res_z': res[2], 
                        'average_distance': average_distance, 'max_distance': max_distance, 'rms_distance': rms_distance, 
                        'average_distance_vb': average_distance, 'max_distance_vb': max_distance_vb, 'rms_distance_vb': rms_distance_vb,
                        'sym_score': dice, 'all_stdv': all_stdv}

            df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

            df.to_csv(output_csv, index=False)


def quantify_symmetry(binary_mask):
    """
    Quantify the symmetry of a binary mask along the y-axis.

    Args:
        binary_mask (np.ndarray): 2D binary mask (0s and 1s).

    Returns:
        float: Symmetry score (0 = no symmetry, 1 = perfect symmetry).
    """
    # Ensure binary_mask is a 2D array
    if binary_mask.ndim != 2:
        raise ValueError("Input must be a 2D binary mask.")
    
    # Determine the midpoint
    height, width = binary_mask.shape
    mid = width // 2
    
    # Split the mask into left and right halves
    left_half = binary_mask[:, :mid]
    right_half = binary_mask[:, mid:]
    
    # If width is odd, remove the central column from the right half
    if width % 2 != 0:
        right_half = right_half[:, 1:]
    
    # Flip the right half horizontally
    right_half_flipped = np.fliplr(right_half)
    
    # Compute the Dice coefficient (2 * |A ∩ B| / (|A| + |B|))
    intersection = np.sum(left_half & right_half_flipped)
    total = np.sum(left_half) + np.sum(right_half_flipped)
    dice_coefficient = (2 * intersection) / total if total > 0 else 0.0
    
    return dice_coefficient


def extract_features_from_df(config: DictConfig):
    '''
    Extract features from the meshes in the parent_dir and save them in the output_dir.
    
    Args:
        input_csv (Path): CSV file containing the paths to the meshes.
        output_csv (Path): CSV file to save the extracted features.

    '''
    input_csv = Path(config.input_csv)
    output_csv = Path(config.output_csv)

    input_csv = pd.read_csv(input_csv)

    if output_csv.exists():
        df = pd.read_csv(output_csv)
    else:
        df = pd.DataFrame(columns=['mesh_path'])
    for i, row in track(input_csv.iterrows(), total=len(input_csv), description=f'Extracting features from {len(input_csv)} cases'):
        mesh = Path(row['mesh_path'])
        if str(mesh) in df['mesh_path'].values:
            log.info(f"Skipping {mesh}")
            continue

        ##### SYMMETRY METRICS #####

        # check that the inferior vector, process and vertebra exist
        vert_mesh_path = mesh.parent / f'{mesh.stem.replace("body_vertebrae", "vertebrae")}.stl'
        vb_mesh_path = mesh
        if not vert_mesh_path.exists():
            log.error(f"Vertebra mesh not found for {mesh}")
            continue
        if not vb_mesh_path.exists():
            log.error(f"Vertebral body mesh not found for {mesh}")

        vert_mesh = pv.read(vert_mesh_path)
        vb_mesh = pv.read(vb_mesh_path)

        plane_point, symmetry_normal = find_plane_of_symmetry(vert_mesh)
        symmetry_plane = pv.Plane(center=plane_point, direction=symmetry_normal, i_size=100, j_size=100)

        # create a plane based on the mesh coordinates [0,0,1]
        frame_normal = np.array([1, 0, 0])
        frame_plane = pv.Plane(center=vb_mesh.center, direction=frame_normal, i_size=100, j_size=100)

        angle = np.arccos(np.dot(symmetry_normal, frame_normal) / (np.linalg.norm(symmetry_normal) * np.linalg.norm(frame_normal)))

        log.debug(f"angle between symmetry normal and frame normal: {np.degrees(angle)}")
        if np.degrees(angle) > 10 and np.degrees(angle) < 160:
            log.debug("Choosing the frame plane")
            chosen_plane = frame_plane
            chosen_normal = frame_normal
        else:
            chosen_plane = symmetry_plane
            chosen_normal = symmetry_normal

        mirroed_mesh, distances = reflect_mesh(vert_mesh, np.array(vb_mesh.center), chosen_normal)

        # Calculate symmetry measures
        average_distance = np.mean(distances)
        max_distance = np.max(distances)
        rms_distance = np.sqrt(np.mean(np.array(distances) ** 2))
        log.debug(f"average_distance: {average_distance}, max_distance: {max_distance}, rms_distance: {rms_distance}")

        try:
            mirroed_mesh, distances = reflect_mesh(vb_mesh, np.array(vb_mesh.center), chosen_normal)
            # Calculate symmetry measures
            average_distance_vb = np.mean(distances)
            max_distance_vb = np.max(distances)
            rms_distance_vb = np.sqrt(np.mean(np.array(distances) ** 2))
            log.debug(f"average_distance: {average_distance_vb}, max_distance: {max_distance_vb}, rms_distance: {rms_distance_vb}")
        except Exception as e:
            log.error(f"Error processing {mesh}: {e}")
            # if the mesh is not empty, display it
            if vb_mesh.n_points > 0:
                p = pv.Plotter()
                p.add_mesh(vb_mesh, color='red')
                p.show()
            continue

        ### HEIGHT METRICS ###

        try:
            heights, binary_mask_high, posterior_angles, inferior_angles, lateral_angles, res = create_heigth_map(vb_mesh_path)
        except Exception as e:
            log.error(f"Error processing {mesh}: {e}")
            continue

        if heights is None:
            log.error(f"Could not create height map for {mesh}")
            continue

        # normalize the heights
        heights = heights / np.max(heights)

        _, s0, s1, s3, s4, s6, s0_a, s0_p, s0_l, s0_r, x_rad_elipse, y_rad_elipse, big_elipse, small_elipse = create_new_sections(heights)

        bin_heights = np.where(heights > 0, 1, 0)
        dice = quantify_symmetry(bin_heights)
        all_stdv = np.std(heights[heights > 0])

        '''
        # Display the height map and the sections
        fig, ax = plt.subplots(2, 4, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing between subplots

        # Row 0
        ax[0, 0].imshow(heights, cmap='viridis')
        ax[0, 0].axis('off')
        ax[0, 0].set_title('Height Map')

        ax[0, 1].imshow(small_elipse, cmap='viridis')
        ax[0, 1].axis('off')
        ax[0, 1].set_title('Small Ellipse')

        ax[0, 2].imshow(s1, cmap='viridis')
        ax[0, 2].axis('off')
        ax[0, 2].set_title('S1')

        ax[0, 3].imshow(s3, cmap='viridis')
        ax[0, 3].axis('off')
        ax[0, 3].set_title('S3')

        # Row 1
        ax[1, 0].imshow(s4, cmap='viridis')
        ax[1, 0].axis('off')
        ax[1, 0].set_title('S4')

        ax[1, 1].imshow(s6, cmap='viridis')
        ax[1, 1].axis('off')
        ax[1, 1].set_title('S6')

        ax[1, 2].imshow(big_elipse, cmap='viridis')
        ax[1, 2].axis('off')
        ax[1, 2].set_title('Big Ellipse')

        # Empty subplot for the last grid cell
        ax[1, 3].axis('off')  # Turn off the axis for the empty subplot

        # Show the figure
        plt.show()
        '''

        # compute the average value for each section including all the values of the height map where the height is bigger than 5 and the section is 255
        s0_avg = np.mean(s0[s0 > 0]) # set 0 to 5 if the heights are not normalized
        s1_avg = np.mean(s1[s1 > 0])
        s3_avg = np.mean(s3[s3 > 0])
        s4_avg = np.mean(s4[s4 > 0])
        s6_avg = np.mean(s6[s6 > 0])
        s0_a_avg = np.mean(s0_a[s0_a > 0])
        s0_p_avg = np.mean(s0_p[s0_p > 0])
        s0_l_avg = np.mean(s0_l[s0_l > 0])
        s0_r_avg = np.mean(s0_r[s0_r > 0])
        ant = s1+s3
        post = s4+s6
        ant_avg = np.mean(ant[ant > 0])
        post_avg = np.mean(post[post > 0])
        center_std = np.std(s0[s0 > 0])
        ant_stdv = np.std(ant[ant > 0])
        post_stdv = np.std(post[post > 0])

        new_row = {'mesh_path': str(mesh), 's0_avg': s0_avg, 's1_avg': s1_avg, 's3_avg': s3_avg, 's4_avg': s4_avg, 
                    's6_avg': s6_avg, 's0_a_avg': s0_a_avg, 's0_p_avg': s0_p_avg, 's0_l_avg': s0_l_avg, 's0_r_avg': s0_r_avg, 
                    'ant_avg': ant_avg, 'post_avg': post_avg, 'center_std': center_std, 'ant_stdv': ant_stdv, 'post_stdv': post_stdv,
                        'W':2*y_rad_elipse, 'posterior_angle_x': posterior_angles[0], 'posterior_angle_y': posterior_angles[1], 
                        'inferior_angle_x': inferior_angles[0], 'inferior_angle_y': inferior_angles[1], 'lateral_angle_x': lateral_angles[0], 
                        'lateral_angle_y': lateral_angles[1], 'res_x': res[0], 'res_y': res[1], 'res_z': res[2], 
                        'average_distance': average_distance, 'max_distance': max_distance, 'rms_distance': rms_distance, 
                        'average_distance_vb': average_distance, 'max_distance_vb': max_distance_vb, 'rms_distance_vb': rms_distance_vb,
                        'sym_score': dice, 'all_stdv': all_stdv}

        df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

        df.to_csv(output_csv, index=False)


