import numpy
import scipy
import scipy.io

def cluster_quality_all(clu, fet, mask, fet_N=12):
    N = clu.shape[0]

    assert N == fet.shape[0], "Number of spikes in fet %r not equal to num_spikes %r" % (f_mask.shape[0], N)

    cluster_ids = numpy.unique(clu)
    unit_quality = numpy.zeros(cluster_ids.shape)
    contamination_rate = numpy.zeros(cluster_ids.shape)
    for i, c in enumerate(cluster_ids):
        clu_idx = numpy.argwhere(clu == c)
        notclu_idx = numpy.argwhere(clu != c)
        n = numpy.size(clu_idx)
        if (n < fet_N) or (n > N / 2):
            unit_quality[i] = 0
            contamination_rate[i] = None

        # use the highest <fet_N> num of mean features for the cluster of interest
        mu_fets = numpy.squeeze(numpy.mean(fet[clu_idx, :], axis=0))
        best_fets = numpy.argsort(-mu_fets, axis=None)
        best_fets = best_fets[0:fet_N - 1]
        # remove the spikes without any features in the best_feature dimension
        cond1 = clu != c
        cond2 = numpy.sum(mask[:, best_fets], axis=1) > 0
        relevant_idx_other = numpy.argwhere(numpy.logical_and(cond1, cond2))

        fet_this = fet[clu_idx, best_fets]
        fet_other = fet[relevant_idx_other, best_fets]

        uQ, cR = cluster_quality_core(fet_this, fet_other)

        unit_quality[i] = uQ
        contamination_rate[i] = cR
        
        print(i+1,' of ',len(cluster_ids))

    return cluster_ids, unit_quality, contamination_rate


def cluster_quality_core(fet_this, fet_other):
    # try:
    n_this = fet_this.shape[0]
    n_other = fet_other.shape[0]

    n_fet = fet_this.shape[1]
    assert n_fet == fet_other.shape[1], "num features dont match"

    cov_this_inv = numpy.linalg.inv(numpy.cov(fet_this, rowvar=False))

    if n_other > n_this and n_this > n_fet:
        md = numpy.zeros(n_other)
        md_self = numpy.zeros(n_this)
        mean_feath_this = numpy.mean(fet_this,axis=0)
        for ii in range(0, n_other):
            md[ii] = numpy.matmul(numpy.matmul(fet_other[ii, :]-mean_feath_this, cov_this_inv), fet_other[ii, :]-mean_feath_this)

        for ii in range(0, n_this):
            md_self[ii] = numpy.matmul(numpy.matmul(fet_this[ii, :]-mean_feath_this, cov_this_inv), fet_this[ii, :]-mean_feath_this)

        md = numpy.sort(md)
        md_self = numpy.sort(md_self)

        unit_quality = md[n_this - 1]
        contamination_rate = 1 - (tipping_point(md_self, md) / numpy.size(md_self))
        #print(tipping_point(md_self, md))
    else:
        unit_quality = 0
        contamination_rate = None
    # except:
        # unit_quality = 0
        # contamination_rate = None

    return unit_quality, contamination_rate


def tipping_point(x, y):
    x = numpy.asarray(x)
    y = numpy.asarray(y)

    n_x = x.shape[0]
    # numpy.concatenate((x, y))
    inds = numpy.argsort(numpy.concatenate((x, y), axis=0))
    inds = numpy.argsort(inds)
    x_inds = inds[0:n_x - 1]

    arr1 = numpy.arange(n_x - 1, 0, -1)
    arr2 = x_inds - numpy.arange(0, n_x - 1, 1)

    pos = numpy.argmax(numpy.less(arr1, arr2))

    if not pos:
        pos = n_x

    return pos
    
if __name__=="__main__":
    # create a small dataset and test the result
    x = numpy.random.normal(0,.1,[1000,12])
    y = numpy.random.normal(2,.1,[10000,12])
        
    uq,cr = cluster_quality_core(x,y)
    print(uq)
    print(cr)
    
    # scipy.io.savemat("C:\\Users\\bsriram\\Desktop\\data.mat",dict(x=x,y=y,md=md,md_self=md_self))