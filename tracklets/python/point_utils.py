import numpy as np
from scipy.linalg import expm3, norm
from scipy.spatial.distance import cdist


def M(axis, theta):
    return expm3(np.cross(np.eye(3), axis / norm(axis) * theta))


def rotate(points, axis, theta):
    if points.ndim == 1:
        return np.squeeze(np.dot(np.expand_dims(points[0:3], axis=0), M(axis, theta)), axis=0)
    else:
        return np.dot(points[:, 0:3], M(axis, theta))


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    Input:
      A: Nx3 numpy array of corresponding 3D points
      B: Nx3 numpy array of corresponding 3D points
    Returns:
      T: 4x4 homogeneous transformation matrix
      R: 3x3 rotation matrix
      t: 3x1 column vector
    '''

    # assert len(A) == len(B)

    # translate points to their centroids

    #centroid_A = np.mean(A, axis=0)
    centroid_A = (np.amax(A, axis=0) - np.amin(A, axis=0)) / 2. + np.amin(A, axis=0)


    centroid_B = np.mean(B, axis=0)
    #AA = A - centroid_A
    #BB = B - centroid_B

    # rotation matrix
    # H = np.dot(AA.T, BB)
    # U, S, Vt = np.linalg.svd(H)
    # R = np.dot(Vt.T, U.T)

    # special reflection case
    # if np.linalg.det(R) < 0:
    #   Vt[2,:] *= -1
    #   R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - centroid_A
    t[2] = 0

    # homogeneous transformation
    T = np.identity(4)
    T[0:3, 3] = t

    return t


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

    zs = src[:,2]
    zc, ze = np.histogram(zs)
    distances = np.empty(src.shape[0], dtype=np.float32)
    indices   = np.empty(src.shape[0], dtype=np.int32)

    for i, p in enumerate(src):
        z = p[2]
        dst_i = np.flatnonzero((dst[:,2] >= (z-0.02)) & (dst[:,2] <= (z+0.02)))
        dst_f = dst[dst_i]
        all_dists = cdist([p], dst_f, 'euclidean')
        index = all_dists.argmin(axis=1)
        distances[i] = all_dists[np.arange(all_dists.shape[0]), index]
        indices[i]   = dst_i[index]


    #all_dists = cdist(src, dst, 'euclidean')
    #indices = all_dists.argmin(axis=1)
    #distances = all_dists[np.arange(all_dists.shape[0]), indices]
    return distances, indices


def icp(A, B, init_pose=None, max_iterations=10000, tolerance=0.0001):
    '''
    The Iterative Closest Point method
    Input:
        A: Nx3 numpy array of source 3D points
        B: Nx3 numpy array of destination 3D point
        init_pose: 4x4 homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation
        distances: Euclidean distances (errors) of the nearest neighbor
    '''

    # make points homogeneous, copy them so as to maintain the originals
    src = np.ones((3, A.shape[0]))
    dst = np.ones((3, B.shape[0]))
    src[0:3, :] = np.copy(A.T)
    dst[0:3, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbours between the current source and destination points
        distances, indices = nearest_neighbor(src[0:3, :].T, dst[0:3, :].T)

        # compute the transformation between the current source and nearest destination points
        t = best_fit_transform(src[0:3, :].T, dst[0:3, indices].T)

        # update the current source
        src += np.expand_dims(t, axis=1) ##np.dot(T, src)

        # check error
        mean_error = np.sum(distances) / distances.size
        if abs(prev_error - mean_error) < tolerance:
            print("yeah", mean_error)
            break
        prev_error = mean_error

    # calculate final transformation
    t = best_fit_transform(A, src[0:3, :].T)

    return t, distances

class ICP(object):

    '''
    2D linear least squares using the hesse normal form:
        d = x*sin(theta) + y*cos(theta)
    which allows you to have vertical lines.
    '''

    def fit(self, first, reference):
        _icp = icp(first, reference)
        return _icp[0]

    def residuals(self, t, first, reference):
        distances, _ = nearest_neighbor(first + t, reference)

        return np.abs(distances)

    def is_degenerate(self, sample):
        return False

def ransac(first, reference, model_class, min_samples, threshold, max_trials=1000):
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

    best_model = None
    best_inlier_num = 0
    best_inliers = None
    best_model_inliers_residua = 1e100
    first_idx = np.arange(first.shape[0])
    import scipy.cluster.hierarchy
    Z = scipy.cluster.hierarchy.linkage(first, 'ward')
    max_d = 3
    clusters = scipy.cluster.hierarchy.fcluster(Z, max_d, criterion='distance')
    unique_clusters = np.unique(clusters)
    print("Unique clusters:", unique_clusters)
    #for _ in xrange(max_trials):
    for cluster in unique_clusters:
        sample_idx = np.where(clusters==cluster)
        sample  = first[sample_idx]
        #sample = first[np.random.randint(0, first.shape[0], np.random.randint(2,min_samples))]
        if model_class.is_degenerate(sample):
            continue
        sample_model = model_class.fit(sample, reference)
        sample_model_residua = model_class.residuals(sample_model, first, reference)
        sample_model_inliers = first_idx[sample_model_residua<threshold]
        inlier_num = sample_model_inliers.shape[0]
        print("Inliers:", inlier_num)
        sample_model_inliers_residua = np.sum(sample_model_residua[sample_model_residua<threshold]) / inlier_num
        if (inlier_num >= min_samples) and (sample_model_inliers_residua < best_model_inliers_residua):
            best_inlier_num = inlier_num
            best_inliers    = sample_model_inliers
            best_model      = sample_model
            best_model_inliers_residua = sample_model_inliers_residua
        print(sample_model_residua)
    #if best_inliers is not None:
    #    best_model = model_class.fit(first[best_inliers], reference)
    return best_model, best_inliers
