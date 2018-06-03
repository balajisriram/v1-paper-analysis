import os
import numpy
import matplotlib.pyplot as plt
import importlib.machinery
import types
import pdb
import pickle


def get_subject_from_session(sess):
    splits = sess.split('_')
    return splits[0]


if __name__=="__main__":
    location = r'C:\Users\bsriram\Desktop\Data_V1Paper\DetailsProcessedPhysOnly'
    
    # create lists
    unit_session = []
    unit_subject = []
    unit_quality = []
    unit_shank = []
    unit_clusterid = []
    for sess in os.listdir(location):
        print(sess)
        with open(os.path.join(location,sess,'spike_and_trials.pickle'),'rb') as f:
            data = pickle.load(f)
        for unit in data['spike_records']['units']:
            if unit['manual_quality'] in ['good','mua']:
                unit_session.append(sess)
                unit_subject.append(get_subject_from_session(sess))
                unit_quality.append(unit['manual_quality'])
                unit_shank.append(unit['shank_no'])
                unit_clusterid.append(unit['cluster_id'])
            elif unit['manual_quality'] in ['unsorted']:
                pass
    
    num_goods = []
    num_mua = []
    
    for sess in os.listdir(location):
        num_goods.append(numpy.sum([x==sess and y in ['good'] for x,y in zip (unit_session,unit_quality)]))
        num_mua.append(numpy.sum([x==sess and y in ['mua'] for x,y in zip (unit_session,unit_quality)]))
    
    num_goods = numpy.asarray(num_goods)
    num_mua = numpy.asarray(num_mua)
    session_avail = numpy.asarray(os.listdir(location))
    
    num_total = numpy.add(num_goods,num_mua)
    fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
    histc, binedge = numpy.histogram(num_total,range=(0,55),bins=11)
    plt.clf()
    plt.bar(binedge[0:11],histc,align='edge',edgecolor='none',color=(0.5,0.5,0.5),width=4)
    plt.show()
    
    fig = plt.figure(figsize=(3,3),dpi=200,facecolor='none',edgecolor='none')
    histc, binedge = numpy.histogram(num_goods,range=(0,55),bins=11)
    plt.clf()
    plt.bar(binedge[0:11],histc,align='edge',edgecolor='none',color=(0.,0.,0.,),width=4)
    plt.show()
        
    pdb.set_trace()
        
        