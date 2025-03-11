import pyvista as pv
import pandas as pd
import numpy as np

# vis_data = pd.read_csv("/home/binigo1/spine_project/extract_vert_feat/results/to_label.csv")
vis_data = pd.read_csv("/home/binigo1/spine_project/extract_vert_feat/results/all_feat_Liver_labeled.csv")
# vis_data = vis_data[vis_data['vertebrae'].str.contains('C')]

for i, row in vis_data.iterrows():
    if row['correct']== 1 or row['correct'] == 0:
        continue
    mesh_path = row['mesh_path']
    vert_mesh_path = mesh_path.replace('body_vertebrae', 'vertebrae')
    p = pv.Plotter()
    vert_mesh = pv.read(vert_mesh_path)
    p.add_mesh(vert_mesh, color='blue')
    p.show()

    # let the user indicate if the segmentation is correct. Append the result to the csv
    correct = input("Is the segmentation correct? (y/n)")
    # add the value to the row where the mesh path is the same
    vis_data.loc[vis_data['mesh_path'] == mesh_path, 'correct'] = (
        1 if correct == 'y' else (0 if correct == 'n' else np.nan)
    )
    vis_data.to_csv("/home/binigo1/spine_project/extract_vert_feat/results/all_feat_Liver_labeled.csv", index=False)