import json
import os
from typing import Any, Dict

import pandas as pd


def str_to_dict(string: str) -> Dict[str, Any]:
    return json.loads(string)


def append_results_to_file(results, filename="results.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if isinstance(results, dict):
        results = {k: [v] for k, v in results.items()}
        results = pd.DataFrame.from_dict(results, orient="columns")
    print(f"Saving results to {filename}")
    # df_pa_table = pa.Table.from_pandas(results)
    if not os.path.isfile(filename):
        results.to_csv(filename, header=True, index=False)
    else:  # it exists, so append without writing the header
        results.to_csv(filename, mode="a", header=False, index=False)
