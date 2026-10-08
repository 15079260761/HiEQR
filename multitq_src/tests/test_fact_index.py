import numpy as np
import pytest

from decgc.retrieval.fact_index import FactIndex


def test_updated_retriever_invalidates_old_index(tmp_path):
    FactIndex(["f"], np.ones((1, 2)), {"retriever_hash": "old"}).save(tmp_path)
    with pytest.raises(RuntimeError, match="stale"):
        FactIndex.load(tmp_path, {"retriever_hash": "new"})
