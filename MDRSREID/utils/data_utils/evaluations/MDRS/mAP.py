import numpy as np
from sklearn.metrics import average_precision_score


def mean_ap(
        dist_mat,
        query_ids=None,
        gallery_ids=None,
        query_cams=None,
        gallery_cams=None,
        average=True):
    """
    Args:
        dist_mat: numpy array with shape [num_query, num_gallery], the
            pairwise distance between query and gallery samples
        query_ids: numpy array with shape [num_query]
        gallery_ids: numpy array with shape [num_gallery]
        query_cams: numpy array with shape [num_query]
        gallery_cams: numpy array with shape [num_gallery]
        average: whether to average the results across queries
    Returns:
        If `average` is `False`:
            ret: numpy array with shape [num_query]
            is_valid_query: numpy array with shape [num_query], containing 0's and
                1's, whether each query is valid or not
        If `average` is `True`:
            a scalar
    """
    # Ensure numpy array
    assert isinstance(dist_mat, np.ndarray)
    assert isinstance(query_ids, np.ndarray)
    assert isinstance(gallery_ids, np.ndarray)
    assert isinstance(query_cams, np.ndarray)
    assert isinstance(gallery_cams, np.ndarray)

    m, n = dist_mat.shape

    # Sort and find correct matches
    indices = np.argsort(dist_mat, axis=1)
    # search the same person images
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = np.zeros(m)
    is_valid_query = np.zeros(m)
    for i in range(m):
        # Filter out the same id and same camera.
        # We not only select the different people images, but also select the same person with different cameras,
        # due to the same person with different cameras are different. We consider these two cases.
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        # this variable will be [True, True, False, False,....]
        # where 'True' means the same person but different cameras,
        # 'False' means the different people.
        y_true = matches[i, valid]
        # '-' means the scores when true should be near zero, false should be away from zero.
        # Thus, the true dists' score should be larger than that of the false ones.
        y_score = -dist_mat[i][indices[i]][valid]
        # If there are no the same person with different cameras, don't calculate the 'aps'.
        if not np.any(y_true): continue
        is_valid_query[i] = 1
        aps[i] = average_precision_score(y_true, y_score)
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    if average:
        return float(np.sum(aps)) / np.sum(is_valid_query)
    return aps, is_valid_query
