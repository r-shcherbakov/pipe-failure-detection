# -*- coding: utf-8 -*-
from typing import Dict, Union, Optional


ALL_TYPES: Dict[str, str] = {
    "datetime": "datetime64[ns]",
    "target": "category",
}

ACCEPTED_BOUNDARIES: Dict[str, Dict[str, float]] = \
    {
    "feature_1": {"min": 0, "max": 25000}, 
    }

FILLNA_CONFIG: Dict[str, Dict[str, Union[Union[float, int], Optional[str]]]] = \
    {  
    "feature_1": {"value": 0, "method": None, "limit": None}, 
    }
    