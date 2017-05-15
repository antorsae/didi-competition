import numpy as np
from scipy.linalg import expm3, norm
import scipy.optimize
from scipy.spatial.distance import cdist

Z_SEARCH_SLICE = 0.02

def M(axis, theta):
    return expm3(np.cross(np.eye(3), axis / norm(axis) * theta))


def rotate(points, axis, theta):
    if points.ndim == 1:
        return np.squeeze(np.dot(np.expand_dims(points[0:3], axis=0), M(axis, theta)), axis=0)
    else:
        return np.dot(points[:, 0:3], M(axis, theta))

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nx3 array of points
        dst: Nx3 array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    distances = np.empty(src.shape[0], dtype=np.float32)
    indices   = np.empty(src.shape[0], dtype=np.int32)

    pending_points  = np.array(src)

    while pending_points.shape[0] > 0:
        z = pending_points[0,2]
        points_in_slice         = np.flatnonzero((src[:,2] > (z-Z_SEARCH_SLICE/2.)) & (src[:,2] < (z+Z_SEARCH_SLICE/2.)))
        pending_points_in_slice = (pending_points[:,2] > (z-Z_SEARCH_SLICE/2.)) & (pending_points[:,2] < (z+Z_SEARCH_SLICE/2.))
        src_slice  = pending_points[pending_points_in_slice]
        dst_i = np.flatnonzero((dst[:,2] >= (z-Z_SEARCH_SLICE)) & (dst[:,2] <= (z+Z_SEARCH_SLICE)))
        dst_f = dst[dst_i]
        all_dists = cdist(src_slice, dst_f, 'euclidean')
        index = all_dists.argmin(axis=1)
        distances[points_in_slice] = all_dists[np.arange(all_dists.shape[0]), index]
        indices[points_in_slice]   = dst_i[index]
        pending_points  = pending_points[~pending_points_in_slice]

    return distances, indices

def rotZ(points, yaw):
    rotMat = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0]])
    return np.dot(points, rotMat)

def norm_nearest_neighbor(t, src, dst, search_yaw=False):
    _t = np.empty(3)
    _t[:2] = t[:2]
    _t[2]  = 0
    if search_yaw:
        yaw    = t[2]
        distances, _ = nearest_neighbor(_t + rotZ(src,yaw), dst)
    else:

        distances, _ = nearest_neighbor(_t + src, dst)
    return np.sum(distances) / src.shape[0]

def icp(A, B, search_yaw=False):
    result = scipy.optimize.minimize(
        norm_nearest_neighbor,
        np.array([0.,0.,3*np.pi/4.-np.pi/2.]) if search_yaw else np.array([0.,0.]),
        args=(A, B, search_yaw),
        method='Powell')
    if search_yaw:
        t = np.empty(4)
        t[:2] = result.x[:2]
        t[2]  = 0
        t[3]  = result.x[2]
        distances, _ = nearest_neighbor(t[:3] + rotZ(A, t[3]), B)
    else:
        t = np.empty(3)
        t[:2] = result.x[:2]
        t[2]  = 0
        distances, _ = nearest_neighbor(t[:3] + A, B)
    return t, distances

class ICP(object):

    '''
    2D linear least squares using the hesse normal form:
        d = x*sin(theta) + y*cos(theta)
    which allows you to have vertical lines.
    '''
    def __init__(self, search_yaw=False):
        self.search_yaw = search_yaw

    def fit(self, first, reference):
        _icp = icp(first, reference, self.search_yaw)
        return _icp[0]

    def residuals(self, t, first, reference):
        if self.search_yaw:
            distances, _ = nearest_neighbor(rotZ(first,t[3]) + t[:3], reference)
        else:
            distances, _ = nearest_neighbor(           first + t[:3], reference)

        return np.abs(distances)


def ransac(first, reference, model_class, min_samples, threshold):
    '''
    Fits a model to data with the RANSAC algorithm.
    :param data: numpy.ndarray
        data set to which the model is fitted, must be of shape NxD where
        N is the number of data points and D the dimensionality of the data
    :param model_class: object
        object with the following methods implemented:
         * fit(data): return the computed model
         * residuals(model, data): return residuals for each data point
         * is_degenerate(sample): return boolean value if sample choice is
            degenerate
        see LinearLeastSquares2D class for a sample implementation
    :param min_samples: int
        the minimum number of data points to fit a model
    :param threshold: int or float
        maximum distance for a data point to count as an inlier
    :param max_trials: int, optional
        maximum number of iterations for random sample selection, default 1000
    :returns: tuple
        best model returned by model_class.fit, best inlier indices
    '''

    min_reference_z =  np.amin(reference[:,2])
    max_reference_z =  np.amax(reference[:,2])

    first_points = first.shape[0]

    # keep points that have a matching slices in the reference object
    first = first[(first[:,2] > (min_reference_z - Z_SEARCH_SLICE)) & (first[:,2] < (max_reference_z + Z_SEARCH_SLICE))]

    if first_points != first.shape[0]:
        print("Removed " + str(first_points - first.shape[0]) + " points due to Z cropping")


    if first.shape[0] > 1:
        print("Fitting " + str(first.shape[0]) + " points to reference object")
    else:
        print("No points to fit, returning")
        return None, None

    best_model = None
    best_inlier_num = 0
    best_inliers = None
    best_model_inliers_residua = 1e100
    second_best_model = None
    second_best_inlier_num = 0
    second_best_inliers = None
    second_best_model_inliers_residua = 1e100
    second_best_score = 0

    first_idx = np.arange(first.shape[0])
    import scipy.cluster.hierarchy
    Z = scipy.cluster.hierarchy.linkage(first, 'single')
    max_d = 0.5
    clusters = scipy.cluster.hierarchy.fcluster(Z, max_d, criterion='distance')
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        sample_idx = np.where(clusters==cluster)
        sample  = first[sample_idx]
        print("Trying cluster " + str(cluster) +  " / " + str(len(unique_clusters)) + " with " + str(sample_idx[0].shape[0]) + " points")

        max_attempts = 10
        while max_attempts > 0:
            sample_model = model_class.fit(sample, reference)
            sample_model_residua = model_class.residuals(sample_model, first, reference)
            sample_model_inliers = first_idx[sample_model_residua<threshold]
            inlier_num = sample_model_inliers.shape[0]
            print("Inliers: " + str(inlier_num) + " / " + str(first_points))
            sample_model_inliers_residua = np.sum(sample_model_residua[sample_model_residua<threshold]) / inlier_num
            if (inlier_num >= min_samples) and (sample_model_inliers_residua < best_model_inliers_residua):
                best_inlier_num = inlier_num
                best_inliers    = sample_model_inliers
                best_model      = sample_model
                best_model_inliers_residua = sample_model_inliers_residua
            elif (inlier_num  / sample_model_inliers_residua) > second_best_score: #(inlier_num >= second_best_inlier_num) and (sample_model_inliers_residua < second_best_model_inliers_residua):
                second_best_score      = inlier_num  / sample_model_inliers_residua
                second_best_inlier_num = inlier_num
                second_best_inliers    = sample_model_inliers
                second_best_model      = sample_model
                second_best_model_inliers_residua = sample_model_inliers_residua

            # keep searching if there's enough inliers and there's other inliers than those
            # used to fit the model
            if (inlier_num < min_samples) or np.all(np.in1d(sample_model_inliers, sample_idx)):
                break
            else:
                sample_idx = sample_model_inliers
                sample = first[sample_idx]
            max_attempts -= 1

    if best_model is not None:
        model, inliers, inlier_num, residua  = best_model, best_inliers, best_inlier_num, best_model_inliers_residua
    else:
        model, inliers, inlier_num, residua =  second_best_model, second_best_inliers, second_best_inlier_num, second_best_model_inliers_residua

    print("Selected " + str(inlier_num) + " / " + str(first_points) + " inliers with " +  str(residua) + " residua")

    return model,inliers


