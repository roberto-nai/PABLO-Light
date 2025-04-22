"""
impressed_pipeline_simple_index.py

Description:

Changelog:
[2024-11-25]: updated pipelines_list = [False] (instead of [True, False])
[2024-11-25]: Added logging.basicConfig() to save application log to file.
[2024-11-25]: Added prefix_col_name in CONF.
[2024-11-25]: Added activity_null in CONF.
[2024-11-25]: Changed neighborhood_size to 1.
[2024-11-25]: Added config.yml and relative files to read it.
[2024-11-25]: Added utilities.dataframe_operations, utilities.general_utilities, utilities.json_operations.
[2024-11-25]: Moved some constant in the config.yml.
[2024-11-27]: Added initial conversion of event log file from XES to CSV.
[2025-01-22]: added dataset_confs.__dict__ to show the dataset configuration used and JSON dump.
[2025-01-26]: added prefix_compute (if 0, the prefix length will be the same as the case length).
[2025-01-26]: added activity_original_position_encoding.
[2025-04-14]: Added compute_activity_positions.
[2025-04-14]: Added retrieve_best_model_v2 function.
"""

### IMPORT ###
import logging
import warnings
# import os
import numpy as np
import pandas as pd
import pm4py
from sklearn.model_selection import train_test_split
import random
from datetime import datetime
import itertools
from sklearn import tree
import json
import dtreeviz
import shutil
from declare4py.declare4py import Declare4Py
from declare4py.enums import TraceState
from pathlib import Path # @RNAI
import re # @RNAI
from pm4py.objects.conversion.log import converter as log_converter # @RNAI

### LOCAL IMPORT ###
from dataset_confs import DatasetConfs
from nirdizati_light.encoding.common import get_encoded_df, EncodingType
from nirdizati_light.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingTypeAttribute
from nirdizati_light.encoding.time_encoding import TimeEncodingType
from nirdizati_light.evaluation.common import evaluate_classifier
from nirdizati_light.explanation.common import ExplainerType, explain
from nirdizati_light.pattern_discovery.common import discovery
from nirdizati_light.hyperparameter_optimisation.common import retrieve_best_model, retrieve_best_model_v2, HyperoptTarget
from nirdizati_light.labeling.common import LabelTypes
from nirdizati_light.log.common import get_log,import_log_csv
from nirdizati_light.predictive_model.common import ClassificationMethods, get_tensor
from nirdizati_light.predictive_model.predictive_model import PredictiveModel, drop_columns
from nirdizati_light.explanation.wrappers.dice_impressed import model_discovery
from nirdizati_light.pattern_discovery.utils.Alignment_Check import alignment_check
import category_encoders as ce
from utilities.dataframe_operations import encode_activities_positions_and_repetitions_v2, encode_activities_positions_and_repetitions_v1, encode_all_activities_positions_and_repetitions_v1, encode_all_activities_positions_and_repetitions_v2, dictionary_unique_values, get_distinct_column_values, find_activity_position_by_index, get_case_lengths, compute_activity_positions, get_label_percentages, plot_feature_importances
from utilities.general_operations import create_directory, create_nested_directory
from utilities.json_operations import extract_data_from_json
from config import config_reader
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.DEBUG,
                    filename="app.log",  # Application log file name
                    filemode="w",        # Overwrite the log file at every exectuion
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )

logger = logging.getLogger(__name__)

### WARNINGS LEVEL ###
# warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")  # @RNAI: Ignore all the warning

### GLOBALS ###
yaml_config = config_reader.config_read_yaml("config.yml", "config")
datasets_dir = str(yaml_config["DATASETS_DIR"])
datasets_sublog_dir = str(yaml_config["DATASETS_SUBLOG_DIR"])
datasets_synth_dir = str(yaml_config["DATASETS_SYNTH_DIR"])
results_dir = str(yaml_config["RESULTS_DIR"])
results_plot_dir = str(yaml_config["RESULTS_PLOT_DIR"])
simple_index_dir = str(yaml_config["SIMPLE_INDEX_DIR"])
process_models_dir = str(yaml_config["PROCESS_MODELS_DIR"])
datasets_position_dir = str(yaml_config["DATASETS_POSITION_DIR"])
datasets_config = str(yaml_config["DATASETS_CONFIG"])
dataset_file_name = str(yaml_config["DATASET_FILE_NAME"])
output_data_dir = str(yaml_config["OUTPUT_DATA"])
prefix_col_name = str(yaml_config["PREFIX_COL_NAME"])
datasets_encoded_dir = str(yaml_config["DATASETS_ENCODED_DIR"])

### results dictionary ###
dic_res = {'datase_name':None, 'prefix_len':None, 'prefix_len':None, 'encoding_name':None, 'activity_name': None, 'sublog_cases': None, 'local_fidelity':None, 'file_encoded': None}
list_res = []

def run_simple_pipeline(CONF=None, dataset_name=None):

    ### RANDOM BASED ON SEED ###
    random.seed(CONF['seed'])
    np.random.seed(CONF['seed'])

    # dataset = CONF['data'].rpartition('/')[0].replace('datasets/','')
    # print(dataset)
    # print(CONF['data'])

    ### Load the dataset (event log) ###
    print(">>>> Extracting dataset")

    logger.debug(f'LOAD DATA - {dataset_name}')

    # Get the dataset format (from file extension)
    file_extension = Path(CONF['data']).suffix[1:]
    print("File format extension:", file_extension.upper())

    # Extract the event log from file
    log = get_log(filepath=CONF['data'])
    # log is a pm4py.objects.log.obj.EventLog instance

    # convert the log into dataframe and CSV (if needed)
    if file_extension.lower() != "csv":
        print("Converting the source event log to Dataframe and CSV")
        log_csv = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
        print("Dataframe shape:", log_csv.shape)
        path_csv = Path(CONF['data']).with_suffix('.csv')
        print("Saving event log into CSV:", path_csv)
        log_csv.to_csv(path_csv, sep=",", index=False)

    logger.debug('Update EVENT ATTRIBUTES')
    print(f"Extracting dataset configuration for '{dataset_name}'")
    dataset_confs = DatasetConfs(dataset_name=dataset_name, where_is_the_file=CONF['data'])
    print(dataset_confs.__dict__) # debug
    file_json = f"{dataset_name}.json"
    path_json = Path(datasets_dir) / dataset_name / file_json
    print("Saving dataset configuration in:", path_json)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(dataset_confs.__dict__, f, ensure_ascii=True, indent=4)
    print()

    print(">>>> Prefix (if needed) and Encoding data")

    max_value = CONF['prefix_length']
    if CONF['prefix_compute'] == 0:
        logger.debug('PREFIX LENGTH SELECTION (if prefix_compute = 0)')
        cases_length = get_case_lengths(log_csv, CONF['case_id_column'])
        max_value = max(cases_length.values(), default=0)
        print("Max case length will be set to:", max_value)
        CONF['prefix_length'] = max_value
        df_case_lengths = pd.DataFrame(cases_length.items(), columns=[CONF['case_id_column'], "case_length"])
        df_case_lengths = df_case_lengths.sort_values(by=CONF['case_id_column'])
        path_cases_csv = Path(datasets_dir) / dataset_name / f"{dataset_name}_case_lengths.csv"
        print("Saving event log case id lengths into CSV:", path_cases_csv)
        df_case_lengths.to_csv(path_cases_csv, sep=',', index=False)

    logger.debug('PREFIX LENGTH: %s', max_value)

    logger.debug('ENCODE DATA')

    encodings = [EncodingType.SIMPLE.value]

    for encoding in encodings:

        CONF['feature_selection'] = encoding
        print("Encoding type:", encoding)

        ### Prefix and encoding ###
        # full_df: full_df contains the complete event log prefix of size CONF['prefix_length'] *with* the activity name encoding (value 0: the i-th activity is not executed)
        # full_df_named: contains the complete event log prefix of size CONF['prefix_length'] *without* the activity name encoding (value 0: the i-th activity is not executed)
        encoder, full_df, full_df_named = get_encoded_df(log=log, CONF=CONF)
        print("Prefix and encoded activities df shape:", full_df.shape)
        # print("Prefix and encoded df columns:",full_df.columns)
        print("Prefix and not encoded activities df shape:", full_df_named.shape)
        # print("Prefix and not encoded df columns:",full_df_named.columns)
        # print("Prefix and not encoded activities df shape:", full_df_named.head())
        
        # Saving prefix encoded (full_df with activities encoded, full_df_named with activities not encoded)
        prefix_suffix = CONF['prefix_length'] if CONF['prefix_compute'] == 1 else "all"

        path_enc = Path(datasets_encoded_dir) / f"{dataset_name}_P-{prefix_suffix}_E-{encoding}_activity_encoded.csv"
        print("Saving encoded dataframe to:", path_enc)
        full_df.to_csv(path_enc, sep = ",", index = False)
        path_enc = Path(datasets_encoded_dir) / f"{dataset_name}_P-{prefix_suffix}_E-{encoding}_activity_named.csv"
        print("Saving encoded (with activity names) dataframe to:", path_enc)
        full_df_named.to_csv(path_enc, sep = ",", index = False)

        # encoder object with data inside (encoded activities)
        # print(encoder._label_dict)
        # print(encoder._label_dict_decoder)

        ### Distinct activities in the generated prefix (obtained from the encoder) ###
        print(">>>> Distinct activities in the generated prefix")
        # full_df / full_df_named
        list_activities_prefix = dictionary_unique_values(encoder._label_dict_decoder, ['label']) # List of strings, ['label'] excluded
        list_activities_prefix_len = len(list_activities_prefix)
        # print(f"Distinct activities found in the prefix ({list_activities_prefix_len}): {list_activities_prefix}") # debug
        print(f"Distinct activities found in the prefix: {list_activities_prefix_len}")

        # Order the list list_activities_prefix
        sorted_activity_names = sorted(list_activities_prefix)
        # Create a dataframe with the distinct atrivities to track them
        df_activities = pd.DataFrame({"activity_num": range(1, len(sorted_activity_names) + 1),"activity_name": sorted_activity_names})
        path_activities = Path(datasets_sublog_dir) / f"{dataset_name}_P-{prefix_suffix}_activity_list.csv"
        print("Saving activities list to:", path_activities)
        df_activities.to_csv(path_activities, sep=',', index=False)
        print()

        ### Predictive model on prefix dataframe - Training ###
        print(">>>> Predictive model - Training")
        logger.debug('TRAIN PREDICTIVE MODEL')
        train_size = CONF['train_val_test_split'][0]
        val_size = CONF['train_val_test_split'][1]
        test_size = CONF['train_val_test_split'][2]
        if train_size + val_size + test_size != 1.0:
            raise Exception('Train-val-test split does not sum up to 1')
        train_df, val_df, test_df = np.split(full_df,[int(train_size*len(full_df)), int((train_size+val_size)*len(full_df))])

        # DOUBT: why this second encoding?
        log_conf = CONF.copy()
        log_conf['feature_selection'] = EncodingType.COMPLEX.value
        complex_encoder, full_df_timestamps, full_df_named = get_encoded_df(log=log, CONF=log_conf)
        train_df_alignment,val_df_alignment,test_df_alignment = np.split(full_df_timestamps,[int(train_size*len(full_df_timestamps)), int((train_size+val_size)*len(full_df_timestamps))])
        complex_encoder.decode(full_df_timestamps)
        complex_encoder.decode(test_df_alignment)

        predictive_model = PredictiveModel(CONF, CONF['predictive_model'], train_df, val_df)
        if CONF['hyperparameter_optimisation']:
            print(">>>> Predictive model - Hyper Tuning")
            logger.debug('HYPERPARAMETER OPTIMISATION')
            predictive_model.model, predictive_model.config = retrieve_best_model(
                predictive_model,
                CONF['predictive_model'],
                max_evaluations=CONF['hyperparameter_optimisation_epochs'],
                target=CONF['hyperparameter_optimisation_target'],seed=CONF['seed']
            )

        ### Predictive model on prefix dataframe - Evaluation ###
        print(">>>> Predictive model - Evaluation")
        logger.debug('EVALUATE PREDICTIVE MODEL')
        if predictive_model.model_type is ClassificationMethods.LSTM.value:
            probabilities = predictive_model.model.predict(get_tensor(CONF, drop_columns(test_df)))
            predicted = np.argmax(probabilities, axis=1)
            scores = np.amax(probabilities, axis=1)
        elif predictive_model.model_type not in (ClassificationMethods.LSTM.value):
            predicted = predictive_model.model.predict(drop_columns(test_df))
            scores = predictive_model.model.predict_proba(drop_columns(test_df))[:, 1]
        actual = test_df['label']
        if predictive_model.model_type is ClassificationMethods.LSTM.value:
            actual = np.array(actual.to_list())
        initial_result = evaluate_classifier(actual, predicted, scores)

        ### Predictive model on prefix dataframe - Explanation ###
        print(">>>> Predictive model - Explanation")
        # model_path = 'results/process_models'
        model_path = Path(results_dir) / process_models_dir
        model_path_str = model_path.as_posix()

        logger.debug('COMPUTE EXPLANATION')

        if CONF['explanator'] is ExplainerType.DICE_IMPRESSED.value:

            print("Explainer type:", ExplainerType.DICE_IMPRESSED.value)

            impressed_pipeline = CONF['impressed_pipeline']

            ### From the test_df (used for the black-box), extracts rows with correct model predictions (based on the dataset can be 0 or 1) ###
            # test_df_correct will be used to generate factual and counter-factual (flipped)
            if 'sepsis_cases_4' in dataset_name:
                test_df_correct = test_df[(test_df['label'] == predicted) & (test_df['label'] == 1)]
            else:
                test_df_correct = test_df[(test_df['label'] == predicted) & (test_df['label'] == 0)]

            method = 'oneshot'
            optimization = 'genetic_conformance'  # 'genetic_conformance' or 'genetic',
            # 'genetic_conformance' considers conformance checking inside the fitness function
            if optimization == 'genetic_conformance':
                adapted = False
            else:
                adapted = None
            diversity = 1.0
            sparsity = 0.5
            proximity = 1.0
            # print(dataset_confs)
            timestamp = [*dataset_confs.timestamp_col.values()][0]
            # neighborhood_size = 75
            ### Neighborhood ###
            neighborhood_size = 1 # @RNAI: 1 from factual, 1 for counter-factual (flipped)
            dynamic_cols = [*dataset_confs.activity_col.values()] + [timestamp]

            ### Sub-log ###
            # For each distinct activity in the list, it extracts all rows (cases) of the dataframe in which the activity appears at least once

            logger.debug(f'Creating sublogs for each activity')

            print(">>>> Sublogs")

            j = 0

            # Remove the null activity from the list of activities to be searched in sublog
            list_activities_prefix = [activity for activity in list_activities_prefix if activity != str(CONF['activity_null'])]

            time_start_sublog = datetime.now() # time to do the sublog tasks

            for activity_name in list_activities_prefix:
                j+=1
                print(f"[{j}] Creating sublog with activity '{activity_name}'")

                if activity_name == str(CONF['activity_null']):
                    print(f"Sublog [{j}] skipped, activity null: {CONF['activity_null']}")
                    continue

                # Identify the columns matching the "prefix_*" pattern
                prefix_columns = [col for col in full_df_named.columns if re.match(r'prefix_\d+$', col)]

                # Identify the case IDs of cases that contain the specified activity in the prefix columns
                # The values in dataframe are forced to str since the list of activities is a list of strings
                case_ids_with_activity = full_df_named[full_df_named[prefix_columns].apply(lambda row: activity_name in row.astype(str).values, axis=1)]['trace_id'].unique()
                case_ids_with_activity_len = len(case_ids_with_activity)
                print(f"Distinct cases containing activity '{activity_name}': {case_ids_with_activity_len}")

                # Filter the dataframe to include only rows belonging to the identified cases
                df_sublog = full_df_named[full_df_named['trace_id'].isin(case_ids_with_activity)]

                # In the sublog, keep the same columns as in full_df (or test_df)
                # print(list(full_df.columns))
                df_sublog = df_sublog[full_df.columns]

                print(f"Sublog [{j}] shape:", df_sublog.shape)
                print(f"Sublog [{j}] distinct cases:", df_sublog['trace_id'].nunique())
                path_sub = Path(datasets_sublog_dir) / f"{dataset_name}_P-{prefix_suffix}_S-{j}_A-{activity_name}.csv"
                print("Saving sublog to:", path_sub)
                df_sublog.to_csv(path_sub, sep = ",", index = False)

                ### Generating a factual and counter-factual (flipped) for every case ###

                # Starting from the test_df_correct (the correct model prediction dataset), generate
                # print(">>>> Generating factual and counter-factual (flipped) for every case of the sublog")
                # print(range(len(test_df_correct.iloc[:50,:])))
                # test_df_correct.iloc[:50,:] -> first 50 rows -> range(0, 50)
                # @RNAI: do query_instance for every entry (case) in sublog
                # for x in range(len(test_df_correct.iloc[:50,:])):

                print(">>>> Generating a factual and counter-factual (flipped) for every case in sublog")
                synth_logs = [] # List with all the syntetic logs generated
                df_sublog_len = len(df_sublog)
                nrows = int(df_sublog_len / CONF["rows_ratio"]) # For testing, divide the number of rows of a ration (1 = all)
                print(f"Rows considered: {nrows} / {df_sublog_len}")

                # Get the case ID of the sublog,
                # extract the row from the dataframe prefix encoded (full_df) as query_instance
                # make counter-factual
                list_cases = get_distinct_column_values(df_sublog, 'trace_id', nrows)
                list_cases_len = len(list_cases)
                print("Distinct cases considered:", list_cases_len)

                x = 0

                logger.debug(f"Creating synthetic sublogs for activity '{activity_name}'")

                for case_id_sublog in list_cases:
                    x+=1
                    print("Query incremental count:", x)

                    # query_instance = test_df_correct.iloc[x, :].to_frame().T # Dataframe with only 1 row and n columns of test_df_correct
                    # query_instance = df_sublog.iloc[x, :].to_frame().T # Dataframe with only 1 row and n columns of test_df_correct
                    query_instance = full_df[full_df['trace_id'] == case_id_sublog].head(1)
                    case_id = query_instance.iloc[0, 0] # extract the case id
                    print("Case ID:", case_id)
                    query_instance = query_instance.drop(columns=['trace_id'])
                    # print("Query instance shape:", query_instance.shape)
                    # Shape of the query_instance and of the full_df should be the same
                    print(f"Query instance dataframe for case ID '{case_id}'") 
                    print(query_instance.head())

                    ### @DOUBT: is it used  in our pipeline???
                    """
                    output_path = 'results/simple_index_results/'
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    discovery_path = output_path+'%s_discovery_%s_%s_%s_%s' % (dataset_name, impressed_pipeline, CONF['seed'],case_id,CONF['prefix_length'])
                    print("Discovery path:", discovery_path) # e.g.  bpic2012_O_ACCEPTED-COMPLETE_discovery_True_48_207518_20
                    """

                    # Drop the trace_id and label columns inside the train_df used in the black-box model
                    columns = drop_columns(train_df).columns
                    # print("Columns")
                    # print(columns)

                    # Generates a list containing all columns in the columns list except those that contain the string ‘Timestamp’ in their name.
                    features_to_vary = [column for column in columns if 'Timestamp' not in column]
                    # print("features_to_vary")
                    # print(features_to_vary)
                    if len(features_to_vary) == 0:
                        features_to_vary = [column for column in columns if 'time' not in column]
                    # print(features_to_vary)

                    # Generates a list containing the column names of a subset of full_df_timestamps that contain the string ‘timestamp’ in their name.
                    timestamps = [col for col in full_df_timestamps.iloc[query_instance.index].columns if 'timestamp' in col]
                    if len(timestamps) == 0:
                        timestamps = [col for col in full_df_timestamps.iloc[query_instance.index].columns if 'Timestamp' in col]
                    # print("timestamps")
                    # print(timestamps)

                    # Generates a new DataFrame df from a subset of rows (query_instance) and columns (timestamps) of full_df_timestamps.
                    df = full_df_timestamps.loc[query_instance.index][timestamps].reset_index()
                    # print("df")
                    # print(df.head(10))
                    
                    timestamps_query = pd.DataFrame(np.repeat(df.values, neighborhood_size*2, axis=0))
                    timestamps_query.columns = df.columns
                    timestamps_query.drop(columns=['index'], inplace=True)
                    # print("timestamps_query")
                    # print(timestamps_query.head(10))

                    # time_start = datetime.now() # time to do the explain task - start
                    # full_df.iloc[:, 1:] -> Selects all rows and all columns, excluding the first column (with index 0)
                    # use test_df instead of full_df

                    # cf_df=full_df.iloc[:, 1:]
                    # print(full_df.head(10))


                    ### Obtains the position of the activity in the case-id to verify conformance in the synthetic log
                    activity_origin_position = find_activity_position_by_index(full_df_named, case_id_sublog, activity_name, "prefix_")
                    print(f"Actual (original) position of activity '{activity_name}': {activity_origin_position} (-1 if not found)")

                    print("--- Generating synthetic log ---")
                    print("Counterfactual method:", method) # oneshot
                    print("Counterfactual optimization:", optimization)
                    synth_log, x_eval, label_list = explain(CONF, predictive_model, encoder=encoder,
                                                            cf_df=full_df.iloc[:, 1:],
                                                            query_instance = query_instance, query_case_id = case_id,
                                                            method=method, optimization=optimization,
                                                            timestamp_col_name=timestamp,
                                                            # model_path = model_path,
                                                            model_path = model_path_str,
                                                            random_seed=CONF['seed'],
                                                            neighborhood_size=neighborhood_size,
                                                            sparsity_weight=sparsity,
                                                            diversity_weight=diversity, proximity_weight=proximity,
                                                            features_to_vary=features_to_vary,
                                                            impressed_pipeline=impressed_pipeline,
                                                            dynamic_cols=dynamic_cols, timestamps=timestamps_query,
                                                            adapted=adapted, activity_origin_position = activity_origin_position, activity_origin_name = activity_name, conformance_penalty = CONF['conformance_penalty'])

                    # Set the name of the case ID following the row index x (Case01 and Case01 are row 0 factual and counter-factual)
                    synth_log['case:concept:name'] = synth_log['case:concept:name'].apply(lambda v: v.replace('Case', f'Case{x}'))
                    print("Synthetic log preview:")
                    print(synth_log.head(5))

                    synth_logs.append(synth_log)
                    print("------")
                    # quit()
                    # time_end = datetime.now()

                    # time_cf = (time_end - time_start).total_seconds() # time_delta to do the explain task - start

                ### Creating an unique synth_log from all the synth cases ###
                print(">>>> Creating an unique synth_log from all the synth cases")
                print("TCOL:", dataset_confs.timestamp_col[dataset_name])
                final_synth_log = pd.concat(synth_logs, ignore_index=True)
                desired_order = [
                "case:concept:name",
                "concept:name",
                # "Complete Timestamp",
                dataset_confs.timestamp_col[dataset_name],
                "case:label",
                "Query_CaseID",
                "likelihood"]

                final_synth_log = final_synth_log[desired_order]

                # final_synth_log = final_synth_log.sort_values(by=['case:concept:name','Query_CaseID'], ascending=True)
                final_synth_log = final_synth_log.sort_values(by=['Query_CaseID'], ascending=True)

                final_synth_log_cases = final_synth_log['Query_CaseID'].nunique() # number of distinct cases in the final synth log

                path_synth = Path(datasets_synth_dir) / f"{dataset_name}_P-{prefix_suffix}_S-{j}_A-{activity_name}_R_{nrows}-{df_sublog_len}_synth.csv"
                print(f"Saving synthetic data of sublog [{j}] to: {path_synth}")
                final_synth_log.to_csv(path_synth, sep = ",", index = False)

                ### 2025-04-14: for each complete syntethic log, get the activity position in the case id ###
                activity_positions = compute_activity_positions(final_synth_log, id_column="case:concept:name", activity=activity_name)
                print(f"Activity '{activity_name}' positions in the synthetic log: {activity_positions}")

                ### Creating an encoding based on activity repetitions from the original prefix dataframe ###
                logger.debug('Creating an encoding based on activity repetitions from the original prefix dataframe')

                print(">>>> Creating an encoding based on activity repetitions from the original prefix dataframe")
                encoding_prefix_act_pos_df = encode_activities_positions_and_repetitions_v2(full_df_named, activity_name)
                # encoding_prefix_act_pos_df = encode_all_activities_positions_and_repetitions_v2(full_df_named)
                path_position = Path(datasets_position_dir) / f"{dataset_name}_P-{prefix_suffix}_S-{j}_A-{activity_name}_R_{nrows}-{df_sublog_len}_origin_pos_enc.csv" # save the encoding of the original dataframe

                # Rename the obtained values (_o for original columns))
                column_rename_mapping = {
                'activity_first': 'activity_first_o',
                'activity_last': 'activity_last_o',
                'activity_repetition': 'activity_repetition_o'
                }

                encoding_prefix_act_pos_df.rename(columns=column_rename_mapping, inplace=True)

                print(f"Saving encoded position of original prefix activities to:", path_position)
                encoding_prefix_act_pos_df.to_csv(path_position, sep = ",", index = False)

                ### Creating an encoding based on activity repetitions from the synthetic prefix dataframe ###

                logger.debug('Creating an encoding based on activity sublog repetition')

                print(">>>> Creating an encoding based on activity sublog repetitions")
                print("Activity query (same as sublog):", activity_name)
                df_attribute_columns = ["likelihood", "Complete Timestamp", "case:label", "Query_CaseID"]
                encoding_synth_act_pos_df = encode_activities_positions_and_repetitions_v1(final_synth_log, activity_name, df_attribute_columns=df_attribute_columns)
                # encoding_synth_act_pos_df = encode_all_activities_positions_and_repetitions_v1(df_in = final_synth_log, df_attribute_columns=df_attribute_columns)

                encoding_synth_act_pos_df = encoding_synth_act_pos_df.sort_values(by=['case:concept:name', 'Query_CaseID'], ascending=False)

                # Columns renaming (_s for synth columns)
                encoding_synth_act_pos_df = encoding_synth_act_pos_df.rename(
                    columns={
                        'activity_first': 'activity_first_s',
                        'activity_last': 'activity_last_s',
                        'activity_repetition': 'activity_repetition_s',
                        'Complete Timestamp': 'Complete Timestamp_s'
                    }
                )

                path_position = Path(datasets_position_dir) / f"{dataset_name}_P-{prefix_suffix}_S-{j}_A-{activity_name}_R_{nrows}-{df_sublog_len}_synth_pos_enc.csv" # save the encoding of the synthetic dataframe
                print(f"Saving encoded position of sublog [{j}] activities to:", path_position)
                encoding_synth_act_pos_df.to_csv(path_position, sep = ",", index = False)

                ### Add the original prefixes to the synthetic ones ###

                # Left-Join the encoding_positions_df with encoding_prefix_df
                join_synth_orig_df = encoding_synth_act_pos_df.merge(encoding_prefix_act_pos_df, how='left', left_on='Query_CaseID', right_on='trace_id')
                path_position = Path(datasets_position_dir) / f"{dataset_name}_P-{prefix_suffix}_S-{j}_A-{activity_name}_R_{nrows}-{df_sublog_len}_join_pos_enc.csv" # save the joint encoding
                print(f"Saving joint position of sublog [{j}] activities to:", path_position)
                join_synth_orig_df.to_csv(path_position, sep = ",", index = False)

                ### DT (white-box) prediction on the joint dataframe ###

                logger.debug('DT (white-box) on joint dataframe')

                testing_percentage = 0.2
                likelihood_mean = join_synth_orig_df['likelihood'].mean()
                col_drop = ['likelihood', 'case:concept:name', 'Complete Timestamp_s', 'Query_CaseID']
                join_synth_orig_df.drop(columns=col_drop,inplace=True)
                join_synth_orig_df.rename(columns={"case:label": "label"}, inplace=True)

                ### Before applying DT, if activity_original_position_encoding is False, use only synth_pos_enc ### 
                if CONF['activity_original_position_encoding'] == 0:
                    join_synth_orig_df = join_synth_orig_df.drop(columns=['activity_first_o', 'activity_last_o', 'activity_repetition_o'])


                ### Get the percentage of labels in the dataset by activity_name ###
                logger.debug(f'Get the percentage of labels in the dataset by activity_name: {activity_name}')
                print(">>>> Get the percentage of labels in the dataset by activity_name")
                label_percentages = get_label_percentages(join_synth_orig_df, 'label')
                print(f"Label percentages (column 'label'):", label_percentages)

                # training, validation, and test starting from the join_synth_orig_df
                # stratify to get same label distribution of the original df

                print("Columns input of the the glass-box model:", join_synth_orig_df.columns)
                print()

                train, test = train_test_split(join_synth_orig_df, test_size=testing_percentage, random_state=42, stratify=join_synth_orig_df['label'])
                train_dt, val_dt = train_test_split(train, test_size=testing_percentage, random_state=42, stratify=train['label'])

                DT_CONF = CONF.copy()
                DT_CONF['predictive_model'] = ClassificationMethods.DT.value
                DT_CONF['hyperparameter_optimisation_target'] = HyperoptTarget.F1.value

                # Ensure that all the columns have name fo string type
                train = train.rename(str, axis="columns")
                test = test.rename(str, axis="columns")
                val_dt = val_dt.rename(str, axis="columns")

                glass_box = PredictiveModel(DT_CONF, DT_CONF['predictive_model'], train_dt, val_dt)
                print("train_dt shape:", train_dt.shape)
                print("train_dt colums:", train_dt.columns)
                print("val_dt shape:", val_dt.shape)
                print("val_dt colums:", val_dt.columns)

                if train_dt.isnull().any().any():
                    print("WARNING: Training data contains NaN values. Check the data.")
                    logger.debug(f'WARNING: Training data contains NaN values. Check the data. Activity {activity_name}')
                    # Handle NaN values (e.g., drop rows, fill with mean, etc.)
                    train_dt = train_dt.dropna()  # Example: drop rows with NaN values
                if val_dt.isnull().any().any():
                    print("WARNING: Validation data contains NaN values. Check the data.")
                    logger.debug(f'WARNING: Training data contains NaN values. Check the data. Activity {activity_name}')
                    # Handle NaN values (e.g., drop rows, fill with mean, etc.)
                    val_dt = val_dt.dropna()  # Example: drop rows with NaN values

                if DT_CONF['hyperparameter_optimisation']:
                        # glass_box.model, glass_box.config = retrieve_best_model_v2(
                        glass_box.model, glass_box.config = retrieve_best_model(
                            glass_box,
                            DT_CONF['predictive_model'],
                            max_evaluations=DT_CONF['hyperparameter_optimisation_epochs'],
                            target=DT_CONF['hyperparameter_optimisation_target'], seed=DT_CONF['seed']
                            )
                        
                if glass_box.model is None:
                    print("WARNING: Hyperparameter optimisation failed. Using default model.")
                    logger.debug(f'WARNING: Hyperparameter optimisation failed. Using default model. Activity {activity_name}')
                    glass_box = PredictiveModel(DT_CONF, DT_CONF['predictive_model'], train_dt, val_dt)
                    glass_box.model = DecisionTreeClassifier(random_state=DT_CONF['seed'])
                    # X_train = train_dt  # Nessuna colonna da escludere
                    # y_train = train_dt['label']
                    # glass_box.model.fit(X_train, y_train)

                glass_box_preds = glass_box.model.predict(drop_columns(test))
                scores = glass_box.model.predict_proba(drop_columns(test))
                local_evaluate_glassbox = evaluate_classifier(test['label'], glass_box_preds, scores)
                local_fidelity = round(local_evaluate_glassbox['accuracy'], 3)
                print('Local fidelity:', round(local_fidelity, 3))

                ### Get DT features importance ###
                features_cols = drop_columns(train_dt).columns
                selected_features = ['activity_first_s', 'activity_last_s', 'activity_repetition_s']
                features_imp = glass_box.model.feature_importances_
                # features_imp is a numpy array with the importance of each feature
                """
                # Create a dictionary mapping each feature to its importance
                feature_importance_mapping = dict(zip(features_cols, np.round(features_imp, 3)))

                # Create the table associating each case:concept:name with the selected feature importances
                records = []

                for concept_name in train_dt['concept:name']: # concept_name is the activity name
                    for feature in selected_features:
                        importance = feature_importance_mapping.get(feature, None)
                        concept_name_position = compute_activity_positions(train_dt, id_column="case:concept:name", activity=concept_name)
                        records.append({
                            "dataset": dataset_name,
                            "dataset_model": f"{dataset_name}_P-{prefix_suffix}_S-{j}_A-{activity_name}_R_{nrows}-{df_sublog_len}_M-{ClassificationMethods.DT.name}",
                            'concept:name': concept_name,
                            'activity_pos_min': concept_name_position['min_pos'],
                            'activity_pos_max': concept_name_position['max_pos'],
                            'activity_pos_avg': concept_name_position['mean_pos'],
                            'feature': feature,
                            'importance': importance
                        })

                feature_importances_per_activity = pd.DataFrame(records)

                # Sort the result
                feature_importances_per_activity = feature_importances_per_activity.sort_values(by=['case:concept:name', 'feature'], ascending=[True, True])
                """
                
                # Get the feature importances (old version for only the specific activity)
                feature_importances_df = pd.DataFrame({
                'dataset_name': dataset_name,
                'cases': df_sublog_len,
                'model': ClassificationMethods.DT.name,
                'preix_compute': CONF['prefix_compute'],
                'prefix_len': CONF['prefix_length'],
                'encoding_name': encoding,
                'activity': activity_name,
                # 'label_percentages': label_percentages,
                'activity_pos_min': activity_positions['min_pos'],
                'activity_pos_max': activity_positions['max_pos'],
                'activity_pos_avg': activity_positions['mean_pos'],
                'feature': features_cols,
                'importance': features_imp,
                'local_fidelity': local_fidelity,
                'file_name': path_position.as_posix()
                })

                feature_importances_per_activity = feature_importances_df.sort_values(by=['feature','importance'], ascending=[True, False])

                ### Save results and timing ###

                time_end_sublog = datetime.now()

                time_delta_sublog_s = round((time_end_sublog- time_start_sublog).total_seconds(),2) # time_delta to do the explain task - start
                time_delta_sublog_m = round((time_end_sublog - time_start_sublog).total_seconds() / 60, 2)  # time_delta in minutes

                print("Delta timing (min):", time_delta_sublog_m)

                print(">>>> Saving the results")

                # Saving DT results
                dic_res = {'datase_name':dataset_name, 'model': ClassificationMethods.DT.name, 'prefix_compute': CONF['prefix_compute'], 'prefix_len': CONF['prefix_length'], 'encoding_name':encoding, 'activity_name': activity_name, 'sublog_cases': final_synth_log_cases, 'likelihood_mean':likelihood_mean, 'local_fidelity_accuracy':local_fidelity, 'sublog_delta_m': time_delta_sublog_m, 'file_encoded_for_glassbox': path_position.as_posix()}
                print(dic_res)

                data_list = [dic_res] # results must be saved in list to be transformed in DataFrame
                df_res = pd.DataFrame(data_list)
                
                # file_res = f"{dataset_name}_P-{prefix_suffix}_S-{j}_A-{activity_name}_R_{nrows}-{df_sublog_len}_M-{ClassificationMethods.DT.name}_results_lfa.csv"
                file_res = f"{dataset_name}_{ClassificationMethods.DT.name}_results_lfa.csv"
                path_res = Path(results_dir) / file_res
                print(f"Path of model '{ClassificationMethods.DT.name}' results:", path_res)
                logger.debug('Saving DT results to:', path_res)
                df_res.to_csv(path_res, mode='w', index=False, header=True)
                if path_res.exists():
                    df_res.to_csv(path_res, mode='a', index=False, header=False)
                else:
                    df_res.to_csv(path_res, mode='w', index=False, header=True)

                # Saving DT features importance
                # file_res = f"{dataset_name}_P-{prefix_suffix}_S-{j}_A-{activity_name}_R_{nrows}-{df_sublog_len}_M-{ClassificationMethods.DT.name}_results_fimp.csv"
                file_res = f"{dataset_name}_{ClassificationMethods.DT.name}_results_fimp.csv"
                path_res = Path(results_dir) / file_res
                print(f"Path of '{ClassificationMethods.DT.name}' feature importance:", path_res)
                logger.debug('Saving DT results feature importance to:', path_res)
                # feature_importances_df.to_csv(path_res, mode='w', index=False, header=True)
                # feature_importances_per_activity.to_csv(path_res, mode='w', index=False, header=True)
                if path_res.exists():
                    feature_importances_per_activity.to_csv(path_res, mode='a', index=False, header=False)
                else:
                    feature_importances_per_activity.to_csv(path_res, mode='w', index=False, header=True)

                logger.debug('DT and feature importance finished')

                ### Plotting DT feature importance ###
                # file_res = f"{dataset_name}_P-{prefix_suffix}_S-{j}_A-{activity_name}_R_{nrows}-{df_sublog_len}_M{ClassificationMethods.DT.name}_results_fimp.png"
                file_res = f"{dataset_name}_{ClassificationMethods.DT.name}_results_fimp.png"
                print("Plotting DT feature importance to:", file_res)
                plot_feature_importances(feature_importances_per_activity, dataset_name, activity_name, results_plot_dir, file_res)

                ### Saving to cumulative results ###
                # Compute the average of the feature importances for the inquiry activity (activity_name)
                # Filter the dataframe for the specified activity
                """
                df_filtered = feature_importances_per_activity[feature_importances_per_activity['activity_name'] == activity_name]

                # Group by 'feature' and compute the mean values
                df_grouped = df_filtered.groupby('feature').agg({
                    'importance': 'mean',
                    'local_fidelity': 'mean'
                }).reset_index()

                # Add the requested columns
                df_grouped['dataset_name'] = dataset_name
                df_grouped['cases'] = df_sublog_len
                df_grouped['model'] = ClassificationMethods.DT.name
                df_grouped['prefix_compute'] = CONF['prefix_compute']
                df_grouped['prefix_len'] = CONF['prefix_length']
                df_grouped['encoding_name'] = encoding
                df_grouped['activity'] = activity_name
                df_grouped['label_percentages'] = label_percentages
                df_grouped['min_pos'] = activity_positions['min_pos'],
                df_grouped['max_pos'] = activity_positions['max_pos'],
                df_grouped['mean_pos'] = activity_positions['mean_pos'],

                file_res = f"{dataset_name}_{ClassificationMethods.DT.name}_results_fimp.csv"
                path_res = Path(results_dir) / file_res
                print(f"Path of '{ClassificationMethods.DT.name}' feature importance:", path_res)
                if path_res.exists():
                    df_grouped.to_csv(path_res, mode='a', index=False, header=False)
                else:
                    df_grouped.to_csv(path_res, mode='w', index=False, header=True)
                """

                logger.debug('RUN IMPRESSED DISCOVERY AND DECISION TREE PIPELINE')
                """
                if impressed_pipeline:
                    discovery_type = 'auto'
                    case_id_col = 'case:concept:name'
                    activity = 'concept:name'
                    outcome = 'case:label'
                    outcome_type = 'binary'
                    delta_time = -1
                    max_gap = CONF['prefix_length'] // 3
                    max_extension_step = 2
                    testing_percentage = 0.2
                    factual_outcome = query_instance['label'].values[0]
                    likelihood = 'likelihood'
                    encoding = True
                    discovery_algorithm = 'impressed'
                    pareto_only = True

                    time_start = datetime.now()
                    train_X, test_X, test_ids = discovery(discovery_algorithm, synth_log, discovery_path, discovery_type, case_id_col, activity, timestamp, outcome,
                              outcome_type, delta_time,
                              max_gap, max_extension_step, factual_outcome, likelihood, encoding,testing_percentage,pareto_only)
                    synth_log.to_csv(discovery_path + '/synthetic_log_%s_%s.csv' % (dataset_name, case_id))

                    time_discovery = (datetime.now() - time_start).total_seconds()

                    pareto_patterns = pd.read_csv(discovery_path + '/paretoset.csv')

                    pareto_patterns['activities'] = pareto_patterns['activities'].str.replace('False',
                                                                                              'directly follows')
                    pareto_patterns['activities'] = pareto_patterns['activities'].str.replace('True',
                                                                                              'eventually follows')
                    try:
                        for idx in range(len(pareto_patterns['activities'])):
                            item = pareto_patterns['activities'].iloc[idx]
                            split_string = item.strip('[').strip(']').strip("'").strip(",").split(',')
                            split_string = split_string[0] + split_string[2].strip(']') +'\n' + split_string[1]
                            pareto_patterns['activities'].iloc[idx] = split_string
                        pareto_patterns.to_csv(discovery_path + '/paretoset.csv')

                    except:
                        'Error in splitting'

                    dict_values = {key: str(value) for key, value in
                                   zip(pareto_patterns['patterns'], pareto_patterns['activities'])}

                    train_X = train_X.rename(columns=dict_values)
                    test_X = test_X.rename(columns=dict_values)

                    synth_log = synth_log.drop(columns=['likelihood'])
                    if 'BPIC17' in dataset_name:
                        synth_log['case:label'].replace({0: 'deviant', 1: 'regular'}, inplace=True)
                    else:
                        synth_log['case:label'].replace({0: 'false', 1: 'true'}, inplace=True)
                    event_log_pred = pm4py.convert_to_event_log(synth_log)
                    _, synth_df = get_encoded_df(log=event_log_pred, CONF=CONF, encoder=encoder)
                    encoder.decode(synth_df)
                    test = synth_df[synth_df['trace_id'].isin(test_ids)]
                    encoder.encode(test)

                    train_X = train_X.rename(str, axis="columns")
                    test_X = test_X.rename(str, axis="columns")
                    train_X = train_X.rename(columns={'Case_ID': 'trace_id', 'Outcome': 'label'})
                    test_X = test_X.rename(columns={'Case_ID': 'trace_id', 'Outcome': 'label'})

                    logger.debug('TRAIN GLASS-BOX MODEL')
                    DT_CONF = CONF.copy()
                    DT_CONF['predictive_model'] = ClassificationMethods.DT.value
                    DT_CONF['hyperparameter_optimisation_target'] = HyperoptTarget.F1.value
                    glass_box = PredictiveModel(DT_CONF, DT_CONF['predictive_model'], train_X, test_X)
                    if DT_CONF['hyperparameter_optimisation']:
                        glass_box.model, glass_box.config = retrieve_best_model(
                            glass_box,
                            DT_CONF['predictive_model'],
                            max_evaluations=DT_CONF['hyperparameter_optimisation_epochs'],
                            target=DT_CONF['hyperparameter_optimisation_target'], seed=DT_CONF['seed']
                        )
                    logger.debug("EVALUATE GLASS-BOX MODEL")
                    glass_box_preds = glass_box.model.predict(np.array(drop_columns(test_X))).astype(int)
                    glass_box_probs = glass_box.model.predict_proba(np.array(drop_columns(test_X)))
                    black_box_preds = predictive_model.model.predict(drop_columns(test))
                    glass_box_result = evaluate_classifier(black_box_preds, glass_box_preds, glass_box_probs)
                    local_fidelity = glass_box_result['accuracy']
                    print('Local fidelity',local_fidelity)

                    logger.debug("EVALUATE GLOBAL GLASS-BOX MODEL")
                    encoder.decode(test_df)
                    test_log_df = pd.wide_to_long(test_df_alignment, stubnames=['prefix', timestamp], i='trace_id',
                                                  j='order', sep='_', suffix=r'\w+').reset_index()
                    test_log_df = test_log_df[dynamic_cols  + ['trace_id','label']]
                    start_alignment = datetime.now()
                    impressed_test_df = alignment_check(log_df=test_log_df,case_id='trace_id',timestamp=timestamp,activity='prefix',
                                                        outcome='label',pattern_folder=discovery_path,delta_time=delta_time)
                    time_alignment = (datetime.now() - start_alignment).total_seconds()

                    impressed_test_df = impressed_test_df.rename(columns=dict_values)

                    impressed_test_df.drop(columns=[timestamp, 'prefix'], inplace=True)
                    impressed_test_df = impressed_test_df[(drop_columns(test_X).columns)]

                    global_preds = glass_box.model.predict(impressed_test_df)
                    global_probs = glass_box.model.predict_proba(impressed_test_df)
                    encoder.encode(test_df)
                    predicted = predictive_model.model.predict(drop_columns(test_df))
                    global_evaluate_glassbox = evaluate_classifier(predicted, global_preds.astype(int), global_probs)
                    global_fidelity = global_evaluate_glassbox['accuracy']
                    print('Global fidelity', global_fidelity)
                    DT_CONF = CONF.copy()
                    DT_CONF['predictive_model'] = ClassificationMethods.DT.value
                    DT_CONF['hyperparameter_optimisation_target'] = HyperoptTarget.F1.value
                    glass_box = PredictiveModel(DT_CONF, DT_CONF['predictive_model'], train_X, test_X)
                    if DT_CONF['hyperparameter_optimisation']:
                        glass_box.model, glass_box.config = retrieve_best_model(
                            glass_box,
                            DT_CONF['predictive_model'],
                            max_evaluations=DT_CONF['hyperparameter_optimisation_epochs'],
                            target=DT_CONF['hyperparameter_optimisation_target'], seed=DT_CONF['seed']
                        )
                    if (local_fidelity > 0.9)  | (global_fidelity > 0.8):
                        viz = dtreeviz.model(glass_box.model,
                                             drop_columns(train_X),
                                             train_X['label'],
                                             feature_names=drop_columns(train_X).columns,
                                             class_names=['false', 'true'],

                                             )
                        v = viz.view(orientation="LR", scale=2, label_fontsize=5.5)
                        v.save(
                            output_path+'decision_trees' + '/' + '%s_impressed_encoding_%s_%s' % (
                            dataset_name, case_id,CONF['prefix_length']) + '.svg')

                    shutil.rmtree(discovery_path)
                else:
                    testing_percentage = 0.2
                    synth_log.drop(columns=['likelihood'],inplace=True)
                    event_log_pred = pm4py.convert_to_event_log(synth_log)
                    frequency_conf = CONF.copy()
                    frequency_conf['feature_selection'] = EncodingType.FREQUENCY.value
                    frequency_encoder, frequency_full_df = get_encoded_df(log=log, CONF=frequency_conf)
                    _, synth_df = get_encoded_df(log=event_log_pred, CONF=frequency_conf,encoder=frequency_encoder)
                    frequency_encoder.decode(synth_df)
                    train, test = train_test_split(synth_df, test_size=testing_percentage, random_state=42,stratify=synth_df['label'])

                    train_dt,val_dt = train_test_split(train, test_size=testing_percentage, random_state=42,stratify=train['label'])
                    DT_CONF = CONF.copy()
                    DT_CONF['predictive_model'] = ClassificationMethods.DT.value
                    DT_CONF['hyperparameter_optimisation_target'] = HyperoptTarget.F1.value
                    train = train.rename(str, axis="columns")
                    test = test.rename(str, axis="columns")
                    train_dt = train_dt.rename(str, axis="columns")
                    val_dt = val_dt.rename(str, axis="columns")

                    glass_box = PredictiveModel(DT_CONF, DT_CONF['predictive_model'], train_dt, val_dt)
                    if DT_CONF['hyperparameter_optimisation']:
                        glass_box.model, glass_box.config = retrieve_best_model(
                            glass_box,
                            DT_CONF['predictive_model'],
                            max_evaluations=DT_CONF['hyperparameter_optimisation_epochs'],
                            target=DT_CONF['hyperparameter_optimisation_target'], seed=DT_CONF['seed']
                        )

                    glass_box_preds = glass_box.model.predict(drop_columns(test))
                    scores = glass_box.model.predict_proba(drop_columns(test))
                    _, synth_df = get_encoded_df(log=event_log_pred, CONF=CONF, encoder=encoder)
                    local_evaluate_glassbox = evaluate_classifier(test['label'], glass_box_preds, scores)
                    local_fidelity = local_evaluate_glassbox['accuracy']
                    print('Local fidelity',local_fidelity)

                    time_discovery = 0
                    time_alignment = 0
                    original_test_df = frequency_full_df[frequency_full_df['trace_id'].isin(test_df['trace_id'])]
                    frequency_encoder.decode(original_test_df)
                    map_keys = [str(k) for k in original_test_df.columns]
                    original_test_df.columns = map_keys
                    original_test_df = original_test_df[drop_columns(test).columns]
                    global_preds = glass_box.model.predict(np.array(original_test_df))
                    global_probs = glass_box.model.predict_proba(original_test_df)
                    global_evaluate_glassbox = evaluate_classifier(predicted, global_preds, global_probs)
                    global_evaluate_actual = evaluate_classifier(predicted, global_preds, global_probs)
                    global_fidelity = global_evaluate_glassbox['accuracy'] if global_evaluate_glassbox['accuracy'] < global_evaluate_actual['accuracy'] else global_evaluate_actual['accuracy']
                    print('Global fidelity', global_fidelity)
                    if (local_fidelity > 0.9) | (global_fidelity > 0.8):
                        viz = dtreeviz.model(glass_box.model,
                                             drop_columns(train),
                                             train['label'],
                                             feature_names=drop_columns(train).columns,
                                             class_names=['false','true'],

                                             )
                        v = viz.view(orientation="LR", scale=2, label_fontsize=5.5)
                        v.save(
                            output_path+'/decision_trees' + '/' + '%s_baseline_%s_%s' % (
                            dataset_name, case_id,CONF['prefix_length'])+'.svg')
                logger.info('RESULT')
                logger.info('Gathering results')
                results = {}
                results['dataset'] = dataset_name
                results['case_id'] = case_id
                results['prefix_length'] = CONF['prefix_length']
                results['impressed_pipeline'] = impressed_pipeline
                results['local_fidelity'] = local_fidelity
                results['global_fidelity'] = global_fidelity
                results['time_discovery'] = time_discovery
                results['time_cf'] = time_cf
                results['time_alignment'] = time_alignment
                results['neighborhood_size'] = neighborhood_size
                results['encoding'] = CONF['feature_selection']
                x_eval['impressed_pipeline'] = impressed_pipeline
                x_eval['extension_step'] = max_extension_step
                try:
                    results['number_of_patterns'] = impressed_test_df.shape[1]
                    results['pareto_only'] = pareto_only
                except:
                    results['number_of_patterns'] = 0
                    results['pareto_only'] = False
                results['extension_step'] = max_extension_step
                results['seed'] = CONF['seed']
                res_df = pd.DataFrame(results, index=[0])
                if not os.path.isfile(output_path+'%s_results_impressed.csv' % (dataset_name)):
                    res_df.to_csv(output_path+'%s_results_impressed.csv' % (dataset_name), index=False)
                else:
                    res_df.to_csv(output_path+'%s_results_impressed.csv' % (dataset_name), mode='a', index=False, header=False)

                try:
                    x_eval['number_of_patterns'] = Impressed_X.shape[1]
                    x_eval['pareto_only'] = pareto_only
                except:
                    x_eval['number_of_patterns'] = 0
                    x_eval['pareto_only'] = False
                x_eval['local_fidelity'] = local_fidelity
                x_eval['global_fidelity'] = global_fidelity
                x_eval = pd.DataFrame(x_eval, index=[0])
                filename_results = output_path+'cf_eval_%s_impressed_%s.csv' % (dataset_name,impressed_pipeline)
                if not os.path.isfile(filename_results):
                    x_eval.to_csv(filename_results, index=False)
                else:
                    x_eval.to_csv(filename_results, mode='a', index=False, header=False)
                print(dataset_name,'impressed_pipeline',impressed_pipeline,
                        'LOCAL FIDELITY',local_fidelity, 'GLOBAL FIDELITY', global_fidelity,'time_discovery',time_discovery,'time_cf',time_cf,
                      'time_alignment',time_alignment,'neighborhood_size',neighborhood_size,'number_of_patterns')
                """

if __name__ == '__main__':
    print()
    print("*** PROGRAM START ***")
    print()

    ### Timing ###
    start_time = datetime.now().replace(microsecond=0)
    print("Start process:", str(start_time))
    print()

    ### Creating output directories ###
    print("> Creating output directories")

    print("Output directory:", datasets_sublog_dir)
    create_directory(datasets_sublog_dir)

    print("Output directory:", datasets_synth_dir)
    create_directory(datasets_synth_dir)

    print("Output directory:", datasets_encoded_dir)
    create_directory(datasets_encoded_dir)

    print("Output directory:", datasets_position_dir)
    create_directory(datasets_position_dir)

    print("Output directory:", results_dir)
    create_directory(results_dir)

    print("Output directory:", results_plot_dir)
    create_directory(results_plot_dir)

    list_dir_results = [results_dir, simple_index_dir]
    print("Output directories:", list_dir_results)
    create_nested_directory(list_dir_results)

    list_dir_results = [results_dir, process_models_dir]
    print("Output directories:", list_dir_results)
    create_nested_directory(list_dir_results)

    print()

    ### Creating output directories ###
    print("> Reading datasets settings")
    path_json = Path(datasets_config)
    print("Path:", path_json)
    datasets_list = extract_data_from_json(path_json)
    datasets_list_len = len(datasets_list)
    # print(f"Datasets found in JSON configuration ({datasets_list_len}):\n", datasets_list, sep="")
    print(f"Datasets found in JSON configuration:", datasets_list_len)
    print()

    if datasets_list_len <= 0:
        print("No dataset informations found, quitting the program")
        quit()

    ### Pipeline ###
    print("> Starting the pipelines")

    # pipelines = [True, False]
    pipelines_list = [False] # @RNAI
    pipelines_list_len = len(pipelines_list)
    print(f"Pipeline list values ({pipelines_list_len}): {pipelines_list}")
    print(f"Datasets that will be parsed: {datasets_list_len}")

    print()

    i = 0 # Dataset counter
    for item in datasets_list:
        i+=1
        print(">> Pipeline on specific dataset")
        print(f"[{i} / {datasets_list_len}]")

        # Get data from JSON
        dataset_name = item['dataset_name']
        dataset_compute = item["dataset_compute"]
        rows_ratio = int(item["rows_ratio"]) # Ratio of lines to be parsed to the total (1 = all)

        prefix_compute =  int(item['prefix_compute'])
        prefix_lengths = item['prefix_list']
        prefix_lengths_len = len(prefix_lengths)

        train_val_test_split = item['train_val_test_split']

        conformance_penalty = float(item['conformance_penalty'])

        activity_original_position_encoding = int(item['activity_original_position_encoding'])

        print("Dataset name:", dataset_name)

        print("Prefix compute (0 = no):", prefix_compute)
        print(f"Dataset prefix lengths ({prefix_lengths_len}):", prefix_lengths)

        print("Dataset rows ratio (1 = all):", rows_ratio)

        print("Conformance penalty:", conformance_penalty)

        path_dataset = Path(datasets_dir) / dataset_name / dataset_file_name 
        print("Dataset path:", path_dataset)

        path_output = Path('..') / output_data_dir
        print("Output path:", path_output)

        if 'bpic2012' in dataset_name:
            seed = 48
        elif 'sepsis' in dataset_name:
            seed = 56
        else:
            seed = 48
        print("Seed:", seed)

        if dataset_compute == 0:
            print(f"Dataset will *not* be processed (dataset_compute = {dataset_compute})")
            print()
            continue
        else:
            print(f"Dataset will be processed (dataset_compute = {dataset_compute})")
            print()
        
        j = 0

        ### Version without prefixes ###
        if prefix_compute == 0:
            print(">> Prefix not computed")
            print(">>> Setting up the pipeline")
            for pipeline in pipelines_list:
                j+=1
                print(f"[{i}.{j} / {datasets_list_len}]")
                CONF = {
                    'data': path_dataset.as_posix(),
                    'train_val_test_split': train_val_test_split,
                    'output': path_output.as_posix(), # @TODO: where is used
                    'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
                    'prefix_length': prefix_compute, # 0 if no prefix are computed (transformed in max possible prefix legth in run_simple_pipeline())
                    'rows_ratio': rows_ratio,
                    'padding': True,  # @TODO: why use of padding?
                    'feature_selection': EncodingType.SIMPLE_TRACE.value,
                    'task_generation_type': TaskGenerationType.ONLY_THIS.value,
                    'attribute_encoding': EncodingTypeAttribute.LABEL.value,  # LABEL, ONEHOT
                    'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
                    'predictive_model': ClassificationMethods.XGBOOST.value,  # RANDOM_FOREST, LSTM, PERCEPTRON
                    'explanator': ExplainerType.DICE_IMPRESSED.value,  # SHAP, LRP, ICE, DICE
                    'threshold': 13,
                    'top_k': 10,
                    'hyperparameter_optimisation': True,
                    'hyperparameter_optimisation_target': HyperoptTarget.AUC.value,
                    'hyperparameter_optimisation_epochs': 20,
                    'time_encoding': TimeEncodingType.NONE.value,
                    'target_event': None,
                    'seed': seed,
                    'impressed_pipeline': pipeline,
                    'prefix_col_name': prefix_col_name,
                    'activity_null': 0,
                    'prefix_compute': prefix_compute,
                    'case_id_column': item['case_id_column'],
                    'conformance_penalty': conformance_penalty,
                    'activity_original_position_encoding': activity_original_position_encoding
                }

                print(">>>> Starting the pipeline")

                run_simple_pipeline(CONF=CONF, dataset_name=dataset_name)

                print()

        j = 0 

        ### Version with prefixes ###
        if prefix_compute == 1:
            print(">> Prefix compute")
            for prefix_length in prefix_lengths:
                print(">>> Setting up the pipeline")
                j+=1
                print(f"[{i}.{j} / {datasets_list_len}]")
                print("Prefix length:", prefix_length)
                for pipeline in pipelines_list:
                    CONF = {
                        'data': path_dataset.as_posix(),
                        'train_val_test_split': train_val_test_split,
                        'output': path_output.as_posix(), # @TODO: where is used
                        'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
                        'prefix_length': prefix_length,
                        'rows_ratio': rows_ratio,
                        'padding': True,  # @TODO: why use of padding?
                        'feature_selection': EncodingType.SIMPLE_TRACE.value,
                        'task_generation_type': TaskGenerationType.ONLY_THIS.value,
                        'attribute_encoding': EncodingTypeAttribute.LABEL.value,  # LABEL, ONEHOT
                        'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
                        'predictive_model': ClassificationMethods.XGBOOST.value,  # RANDOM_FOREST, LSTM, PERCEPTRON
                        'explanator': ExplainerType.DICE_IMPRESSED.value,  # SHAP, LRP, ICE, DICE
                        'threshold': 13,
                        'top_k': 10,
                        'hyperparameter_optimisation': True,
                        'hyperparameter_optimisation_target': HyperoptTarget.AUC.value,
                        'hyperparameter_optimisation_epochs': 20,
                        'time_encoding': TimeEncodingType.NONE.value,
                        'target_event': None,
                        'seed': seed,
                        'impressed_pipeline': pipeline,
                        'prefix_col_name': prefix_col_name,
                        'activity_null': 0,
                        'prefix_compute': prefix_compute,
                        'activity_original_position_encoding': activity_original_position_encoding
                    }

                    print(">>>> Starting the pipeline")

                    run_simple_pipeline(CONF=CONF, dataset_name=dataset_name)

                    print()
        print()

    end_time = datetime.now().replace(microsecond=0)
    print("End process:", str(end_time))
    delta_time =  end_time - start_time
    delta_min = round(delta_time.total_seconds() / 60 , 2)
    delta_sec = delta_time.total_seconds()
    print("Time to process:", str(delta_time))
    print("Time to process in min:", str(delta_min))
    print("Time to process in sec:", str(delta_sec))
    print()
    
    logger.debug(f"End process: {str(end_time)}")
    logger.debug(f"Time to process: {str(delta_time)}")

    print()
    print("*** PROGRAM END ***")
    print()