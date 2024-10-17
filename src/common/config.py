from typing import Dict, Union, Optional

from common.features import MANDATORY_FEATURES


FEATYPE_TYPES: Dict[str, str] = {feature.name: feature.dtype for feature in MANDATORY_FEATURES}

CLIP_CONFIG: Dict[str, Dict[str, float]] = \
    {feature.name: {"lower": feature.lower, "upper": feature.upper} for feature in MANDATORY_FEATURES}

FILLNA_CONFIG: Dict[str, Dict[str, Union[Union[float, int], Optional[str]]]] = \
    {feature.name:
        {"value": feature.fillna_value, "method": feature.fillna_method, "limit": feature.fillna_limit}
        for feature in MANDATORY_FEATURES
    }
