"""
common.py

Description: .

Changelog:
[2024-11-25]: Added query_case_id parameter to the method explain.
[2024-11-25]: Added query_case_id parameter to the method dice_impressed.
"""

from enum import Enum

from nirdizati_light.explanation.wrappers.dice_impressed import dice_impressed
class ExplainerType(Enum):
    DICE_IMPRESSED = 'dice_impressed'

def explain(CONF, predictive_model, encoder, cf_df=None,
            method=None, optimization=None, support=0.9, timestamp_col_name=None,
            model_path=None,random_seed=None,query_instance=None, query_case_id=None, neighborhood_size=None,
            diversity_weight=None,sparsity_weight=None,proximity_weight=None,features_to_vary=None,
            impressed_pipeline=None,dynamic_cols=None,timestamps=None, adapted=None):
    explainer = CONF['explanator']
    if explainer is ExplainerType.DICE_IMPRESSED.value:
        return dice_impressed(CONF, predictive_model, encoder=encoder, cf_df=cf_df,
                              query_instance=query_instance,
                              query_case_id = query_case_id,
                              method=method, optimization=optimization,
                              support=support, timestamp_col_name=timestamp_col_name,
                              model_path=model_path,
                              random_seed=random_seed, neighborhood_size=neighborhood_size,
                              sparsity_weight=sparsity_weight, diversity_weight=diversity_weight
                              , proximity_weight=proximity_weight, features_to_vary=features_to_vary,
                              impressed_pipeline=impressed_pipeline,
                              dynamic_cols=dynamic_cols, timestamps=timestamps, adapted=adapted)
