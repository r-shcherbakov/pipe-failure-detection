from common.constants import SECONDS_IN_MINUTE

DEFAULT_CONFIG = {
    "feature_a_sub_median_2_shift_15": {
        "feature_source": "feature_a",
        "transformers": [
            {'transform_name': 'PositiveReplacer',
                "config": {
                    "pos_value": 3, 
                    }
                },
            {'transform_name': 'OutlierImputer'},
            {'transform_name': 'FourierTransformer',
                "config": {
                    "threshold": 3, 
                    }
                },
            {"transform_name": "Converter",
                "config": {
                    "method": "substract",
                    "first_window": SECONDS_IN_MINUTE * 2,
                    "first_shift_size": 0,
                    "first_agg_func": "median",
                    "second_window": SECONDS_IN_MINUTE * 2,
                    "second_shift_size": SECONDS_IN_MINUTE * 15, 
                    "second_agg_func": "median",
                    },
                },
            {'transform_name': 'FourierTransformer'},
            ],
        },
    
    }
