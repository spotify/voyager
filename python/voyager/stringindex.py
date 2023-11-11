from typing import Dict
import numpy as np
from voyager import Index

INDEX_FILE_NAME: str = "index.hnsw"
NAMES_LIST_FILE_NAME: str = "names.json"

class StringIndex:
    """
    A wrapper class around an Index with a simplified interface which maps the index ID to a provided String.
    """
    index: Index
    reverse_id_lookup: Dict[int, str]

    def __init__(self, *args, **kwargs):
        """
        Construct a StringIndex.  Any constructor args are automatically passed through to Index construction.
        :param args:
        :param kwargs:
        """
        self.index = Index(*args, **kwargs)
        self.reverse_id_lookup = {}
        self.insertion_id = 0

    def query(
        self,
        vectors: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Given a voyager index and set of query vectors, perform an NN lookup and translate the results back into id strings
        """
        try:
            voyager_ids, _ = self.index.query(vectors, **kwargs)
            # Convert voyager IDs into a numpy array and then perform a lookup in the reverse_id_lookup for
            # each item in order to convert it back to its external string ID
            ids_np = np.array(voyager_ids, dtype="int32")
            result_ids = np.vectorize(self.reverse_id_lookup.get)(voyager_ids)
            return result_ids
        except IndexError:
            return np.array([])

    def add_items(self, string_ids, vectors, *args, **kwargs):
        inserted_ids = self.index.add_items(vectors, *args, **kwargs)
        to_update = {insertion_id: string_id for insertion_id, string_id in zip(inserted_ids, string_ids)}
        self.reverse_id_lookup.update(to_update)



