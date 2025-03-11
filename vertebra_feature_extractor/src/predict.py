import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/home/binigo1/spine_project/rulefit')
from rulefit.rulefit import RuleFit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import logging
from pathlib import Path
from omegaconf import DictConfig
from rich.logging import RichHandler
from rich.progress import track

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger(__name__)

def postprocess(results: pd.DataFrame) -> pd.DataFrame:
    '''
    Postprocess the results of the feature extraction
    '''
    results.drop_duplicates(subset=['mesh_path'], inplace=True)

    results['posterior_angle_x'] = results['posterior_angle_x'].apply(lambda x: x if x < 90 else x-90)
    results['posterior_angle_y'] = results['posterior_angle_y'].apply(lambda x: x if x < 90 else x-90)
    results['inferior_angle_x'] = results['inferior_angle_x'].apply(lambda x: x if x < 90 else x-90)
    results['inferior_angle_y'] = results['inferior_angle_y'].apply(lambda x: x if x < 90 else x-90)
    results['lateral_angle_x'] = results['lateral_angle_x'].apply(lambda x: x if x < 90 else x-90)
    results['lateral_angle_y'] = results['lateral_angle_y'].apply(lambda x: x if x < 90 else x-90)
    # if the angles are bigger than 45, subtract 90 and take the absolute value
    results['posterior_angle_x'] = results['posterior_angle_x'].apply(lambda x: 90-x if x > 45 else x)  
    results['posterior_angle_y'] = results['posterior_angle_y'].apply(lambda x: 90-x if x > 45 else x)
    results['inferior_angle_x'] = results['inferior_angle_x'].apply(lambda x: 90-x if x > 45 else x)
    results['inferior_angle_y'] = results['inferior_angle_y'].apply(lambda x: 90-x if x > 45 else x)
    results['lateral_angle_x'] = results['lateral_angle_x'].apply(lambda x: 90-x if x > 45 else x)
    results['lateral_angle_y'] = results['lateral_angle_y'].apply(lambda x: 90-x if x > 45 else x)

    results['vertebrae'] = results['mesh_path'].apply(lambda x: x.split('/')[-1].split('_')[-1].split('.')[0])
    vert_dict = {1:'C1', 2:'C2', 3:'C3', 4:'C4', 5:'C5', 6:'C6', 7:'C7', 8:'T1', 9:'T2', 10:'T3', 11:'T4', 12:'T5', 13:'T6', 14:'T7', 15:'T8', 16:'T9', 17:'T10', 18:'T11', 19:'T12', 20:'L1', 21:'L2', 22:'L3', 23:'L4', 24:'L5', 25:'S1'}
    inverse_vert_dict = {v: k for k, v in vert_dict.items()}
    results['vertebrae_#'] = results['vertebrae'].apply(lambda x: inverse_vert_dict[x])
    results['case'] = results['mesh_path'].apply(lambda x: x.split('/')[-3])

    for i, row in results.iterrows():    
        rel_cent_ant = row['s0_avg'] / row['ant_avg']
        rel_cent_post = row['s0_avg'] / row['post_avg']
        rel_ant_post = row['ant_avg'] / row['post_avg']

        results.loc[i, 'rel_cent_ant'] = rel_cent_ant
        results.loc[i, 'rel_cent_post'] = rel_cent_post
        results.loc[i, 'rel_ant_post'] = rel_ant_post

        rel_s0_l_r = row['s0_l_avg'] / row['s0_r_avg']
        rel_s0_a_p = row['s0_a_avg'] / row['s0_p_avg']
        rel_s0_p_ant = row['s0_p_avg'] / row['ant_avg']
        rel_s0_a_post = row['s0_a_avg'] / row['post_avg']

        results.loc[i, 'rel_s0_l_r'] = rel_s0_l_r
        results.loc[i, 'rel_s0_a_p'] = rel_s0_a_p
        results.loc[i, 'rel_s0_p_ant'] = rel_s0_p_ant
        results.loc[i, 'rel_s0_a_post'] = rel_s0_a_post

    results["rat_x_y"] = results["res_x"]/results["res_y"]
    results["rat_y_z"] = results["res_y"]/results["res_z"]
    results["rat_x_z"] = results["res_x"]/results["res_z"]

    # drop nan values
    results.dropna(inplace=True)

    return results

def find_balanced_threshold(precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray) -> float:
    """
    Find the threshold where precision and recall are closest (balanced).

    Parameters:
        precision (numpy.ndarray): Array of precision values.
        recall (numpy.ndarray): Array of recall values.
        thresholds (numpy.ndarray): Array of threshold values.

    Returns:
        balanced_threshold (float): The threshold where precision and recall are balanced.
    """
    # Compute the absolute difference between precision and recall
    differences = np.abs(precision - recall)
    
    # Find the index of the minimum difference
    balanced_index = np.argmin(differences)
    
    # Return the corresponding threshold
    return thresholds[balanced_index]


def extract_rules(labeled_data: pd.DataFrame) -> dict:
    '''
    Finds the sets of rules that optimizes both precision and recall for every vertebrae group
    '''
    if np.any(labeled_data['correct'].isin(['y', 'n'])):
        labeled_data['correct'] = labeled_data['correct'].map({'y': 1, 'n': 0})
    
    assert labeled_data['correct'].dtype == int or labeled_data['correct'].dtype == float, f"The correct column must be a class. Found {labeled_data['correct'].unique()}"

    vertebrae_groups = {'C1_C7': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'], 'T1_T4': ['T1', 'T2', 'T3', 'T4'],
                        'T5_T8': ['T5', 'T6', 'T7', 'T8'], 'T9_T12': ['T9', 'T10', 'T11', 'T12'], 'L1_L5': ['L1', 'L2', 'L3', 'L4', 'L5']}
    vertebrae_groups = {'C1-T3': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7','T1', 'T2', 'T3'], 'T4_T12': ['T4','T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'], 'L1_L5': ['L1', 'L2', 'L3', 'L4', 'L5']}
    vertebrae_groups = {'cervix': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'], 'thorax': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'], 'lumbar': ['L1', 'L2', 'L3', 'L4', 'L5']}
    rules = {}

    for group_name, vertebrae in vertebrae_groups.items():
        group_data = labeled_data[labeled_data['vertebrae'].isin(vertebrae)]
        group_data = group_data.copy()
        group_data.drop(["vertebrae", "vertebrae_#", "case"], axis=1, inplace=True)
        train, test = train_test_split(group_data, test_size=0.3, random_state=42)

        train_feat = train.drop(columns=['mesh_path'])
        test_feat = test.drop(columns=['mesh_path'])

        X = train_feat.drop('correct', axis=1)
        y = train_feat['correct'].values
        y = y.astype(int)
        features = X.columns
        X = np.array(X)

        gc = GradientBoostingClassifier(n_estimators=5000, max_depth=100, learning_rate=0.01, random_state=42)
        rf = RuleFit(tree_generator=gc, rfmode="classify", max_rules=10, random_state=0, model_type='lr')
        rf.fit(X, y, feature_names=features)

        model = rf.predict(X)[0]
        new_features = rf.predict(X)[1]
        prob = rf.predict_proba(X)

        precision, recall, ths = precision_recall_curve(y, prob[:, 1])
        balanced_threshold = find_balanced_threshold(precision, recall, ths)

        X = test_feat.drop('correct', axis=1)
        y = test_feat['correct'].values
        X = np.array(X)

        model = rf.predict(X)[0]
        new_features = rf.predict(X)[1]
        prob = rf.predict_proba(X)

        test['predicted'] = prob[:, 1] > balanced_threshold

        # save the confusion matrix (normalized and unnormlized)
        cm = confusion_matrix(test['correct'], test['predicted'])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Replace NaNs (if any due to division by zero)
        cm_norm = np.nan_to_num(cm_norm)

        # save both the normalized and unnormlized confusion matrix as one unique png
        test_cm_folder = Path('test_CM')
        test_cm_folder.mkdir(exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sns.heatmap(cm, annot=True, ax=ax[0], fmt='g', cmap='Blues')
        ax[0].set_title('Unnormalized confusion matrix')
        sns.heatmap(cm_norm, annot=True, ax=ax[1], fmt='.2f', cmap='Blues')
        ax[1].set_title('Normalized confusion matrix')
        plt.tight_layout()
        plt.savefig(f'{test_cm_folder}/confusion_matrix_{group_name}.png')
        plt.close()
        log.info(f"Saved confusion matrix for {group_name} to confusion_matrix_{group_name}.png")

        rules[group_name] = {'vertebrae': vertebrae, 'rules': rf, 'balanced_threshold': balanced_threshold}

    return rules


def predict_quality_seg(config: DictConfig):
    '''
    train data must contain the labels for "correct" and the features for the training
    inference data must contain the features for the inference
    '''
    train_data_path = Path(config.train_data)
    inference_data_path = Path(config.inference_data)

    train_data = pd.read_csv(train_data_path)
    inference_data = pd.read_csv(inference_data_path)

    train_data = postprocess(train_data)
    inference_data = postprocess(inference_data)

    log.info("Extracting rules")
    rules = extract_rules(train_data)

    inference_predictions = pd.DataFrame(columns=["mesh_path", "predicted"])

    # iterate throught the dict of rules and predict the inference data
    for group_name, model in track(rules.items(), description=f"Predicting {len(rules)} groups"):
        vertebrae = model['vertebrae']
        log.info(f"Predicting {group_name}")
        rf = model['rules']
        balanced_threshold = model['balanced_threshold']

        inference_data_group = inference_data[inference_data['vertebrae'].isin(vertebrae)]
        # check that the dataset is not empty
        if inference_data_group.empty:
            log.error(f"No data found for {group_name}")
            continue

        X = inference_data_group.drop(columns=['mesh_path', 'vertebrae', 'vertebrae_#', 'case'])
        features = X.columns
        X = np.array(X)

        prob = rf.predict_proba(X)

        inference_data_group = inference_data_group.copy()
        inference_data_group.loc[:, 'predicted'] = prob[:, 1] > balanced_threshold


        inference_predictions_group = inference_data_group[['mesh_path', 'predicted']]
        inference_predictions = pd.concat([inference_predictions, inference_predictions_group])

    output_file = str(inference_data_path.parent / inference_data_path.stem) + '_predictions.csv'
    log.info(f"Saving predictions to {output_file}")
    inference_predictions.to_csv(output_file, index=False)


        
        



        

            

