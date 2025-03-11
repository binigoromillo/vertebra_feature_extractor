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
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import sys
from tqdm import tqdm

from rich.logging import RichHandler
from rich.progress import track

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger(__name__)

from .features import *

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


def vis_features(config: DictConfig):
    '''

    '''
    csv = pd.read_csv(config.vis_data)
    csv = csv[csv['correct'] == 0]
    meshes = csv['mesh_path'].values
    for mesh in meshes:
        mesh = Path(mesh)
        # check that the inferior vector, process and vertebra exist
        vert_mesh_path = mesh.parent / f'{mesh.stem.replace("body_vertebrae", "vertebrae")}.stl'
        inferior_vector_path = mesh.parent / f'{mesh.stem.replace("body_vertebrae", "normal_SI_vertebrae")}.npy'
        process_mesh_path = mesh.parent / f'{mesh.stem.replace("body_vertebrae", "processes_vertebrae")}.stl'
        vb_mesh_path = mesh
        if not vert_mesh_path.exists():
            log.error(f"Vertebra mesh not found for {mesh}")
            continue
        if not inferior_vector_path.exists():
            log.error(f"Inferior vector not found for {mesh}")
            continue
        if not vb_mesh_path.exists():
            log.error(f"Vertebral body mesh not found for {mesh}")
            continue
        if not process_mesh_path.exists():
            log.error(f"Process mesh not found for {process_mesh_path}")
            continue

        vert_mesh = pv.read(vert_mesh_path)
        inferior_vector = np.load(inferior_vector_path) # axial normal
        vb_mesh = pv.read(vb_mesh_path)
        process_mesh = pv.read(process_mesh_path)

        process_mesh = find_optimal_process_CoM(process_mesh, vb_mesh, inferior_vector)

        # recompute the saggital normal
        coronal_normal = np.array(vb_mesh.center) - np.array(process_mesh.center)
        saggital_normal = np.cross(coronal_normal, inferior_vector)
        saggital_plane = pv.Plane(center=vb_mesh.center, direction=saggital_normal, i_size=100, j_size=100)

        plane_point, symmetry_normal = find_plane_of_symmetry(vert_mesh)
        symmetry_plane = pv.Plane(center=plane_point, direction=symmetry_normal, i_size=100, j_size=100)

        # create a plane based on the mesh coordinates [0,0,1]
        frame_normal = np.array([1, 0, 0])
        frame_plane = pv.Plane(center=vb_mesh.center, direction=frame_normal, i_size=100, j_size=100)

        angle = np.arccos(np.dot(symmetry_normal, frame_normal) / (np.linalg.norm(symmetry_normal) * np.linalg.norm(frame_normal)))

        log.info(f"angle between symmetry normal and frame normal: {np.degrees(angle)}")
        if np.degrees(angle) > 10 and np.degrees(angle) < 160:
            log.info("Choosing the frame plane")
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
        log.info("WHOLE MESH")
        log.info(f"average_distance: {average_distance}, max_distance: {max_distance}, rms_distance: {rms_distance}")

        p = pv.Plotter(off_screen=False)
        p.add_mesh(vert_mesh, color='g', opacity=0.3)
        # p.add_mesh(frame_plane, color='r', opacity=0.5)
        p.add_mesh(mirroed_mesh, color='r', opacity=0.5)
        p.add_mesh(chosen_plane, color='b', opacity=0.5)
        # p.add_mesh(saggital_plane, color='c', opacity=0.5)
        p.show()

def vis_inspection(config: DictConfig):
    """
    Visualizes meshes for inspection and updates a CSV file to record the correctness of segmentations.

    Args:
        config (DictConfig): Configuration object containing paths and settings.
    """
    prediction = config.label
    vertebrae_types = config.vertebrae

    # Load the data
    csv = pd.read_csv(config.vis_data)
    csv_filtered = csv[csv['predicted'] == prediction]  # Filter by prediction
    meshes = csv_filtered['mesh_path'].values

    vertebrae = [f"{vert}.stl" for vert in vertebrae_types]
    log.info(vertebrae)
    meshes = [mesh for mesh in meshes if any(vert in mesh for vert in vertebrae)]

    # meshes = [mesh for mesh in meshes if 'NMDID' not in mesh]

    csv_filtered = csv_filtered[csv_filtered['mesh_path'].isin(meshes)]
    print(f"Total meshes {len(meshes)}")
    
    # check if there is a field called "vis_ground_truth" in the config file
    if config.vis_ground_truth:
        log.info("Visualizing ground truth for {} segmentations".format(prediction))
        csv_filtered = csv[csv['correct'] == prediction]
        meshes = csv_filtered['mesh_path'].values
        np.random.seed(0)
        np.random.shuffle(meshes)
        meshes = meshes[:100]
        
        print(f"Total meshes to visualize: {len(meshes)}")

        # Loop through meshes and visualize
        for mesh in tqdm(meshes, desc="Visualizing"):
            mesh_path = Path(mesh)

            # Check that the corresponding vertebrae mesh exists
            vert_mesh_path = mesh_path.parent / f"{mesh_path.stem.replace('body_vertebrae', 'vertebrae')}.stl"
            if not vert_mesh_path.exists():
                log.warning(f"Missing vertebrae mesh: {vert_mesh_path}")
                continue
            log.info(f"Visualizing {mesh}")

            # Load and visualize the mesh
            vert_mesh = pv.read(vert_mesh_path)
            p = pv.Plotter()
            p.add_mesh(vert_mesh, color="w", opacity=0.7)
            p.show()
                
    else:
        np.random.seed(0)
        np.random.shuffle(meshes)
        meshes = meshes[:100]
        
        # check if any of the rows have nan in the correct column
        if csv[csv['mesh_path'].isin(meshes)]['correct'].isnull().values.any():

            log.info(f"Visualizing meshes predicted as {prediction} for {vertebrae_types}")
            print(f"Total meshes to visualize: {len(meshes)}")

            # Loop through meshes and visualize
            for mesh in tqdm(meshes, desc="Visualizing"):
                mesh_path = Path(mesh)

                # Check that the corresponding vertebrae mesh exists
                vert_mesh_path = mesh_path.parent / f"{mesh_path.stem.replace('body_vertebrae', 'vertebrae')}.stl"
                if not vert_mesh_path.exists():
                    log.warning(f"Missing vertebrae mesh: {vert_mesh_path}")
                    continue
                
                if "correct" in csv.columns:
                    if csv.loc[csv['mesh_path'] == str(mesh_path), 'correct'].values[0] == 1 or csv.loc[csv['mesh_path'] == str(mesh_path), 'correct'].values[0] == 0:
                        log.info("Segmentation already marked")
                        continue
                log.info(f"Visualizing {mesh}")

                # Load and visualize the mesh
                vert_mesh = pv.read(vert_mesh_path)
                p = pv.Plotter()
                p.add_mesh(vert_mesh, color="w", opacity=0.7)
                p.show()

                # Collect user input and update the DataFrame
                correct = input("Is the segmentation correct? (y/n): ").strip().lower()
                value = True if correct == 'y' else (False if correct == 'n' else np.nan)

                # Update the 'correct' column for the corresponding row
                csv.loc[csv['mesh_path'] == str(mesh_path), 'correct'] = value

                # Save the updated DataFrame
                csv.to_csv(config.vis_data, index=False)
            log.info(f"Updated CSV saved to {config.vis_data}")
        else:
            log.info("All segmentations have been marked as correct or incorrect.")
            log.info(f"# True positives: {len(csv[(csv['correct'] == 1) & (csv['predicted'] == prediction)])}")
            log.info(f"# False positives: {len(csv[(csv['correct'] == 0) & (csv['predicted'] == prediction)])}")
            log.info(f"# True negatives: {len(csv[(csv['correct'] == 0) & (csv['predicted'] != prediction)])}")
            log.info(f"# False negatives: {len(csv[(csv['correct'] == 1) & (csv['predicted'] != prediction)])}")

       