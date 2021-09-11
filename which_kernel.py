"""
Information about kernels, somewhat similar to (ck)brief toolkit. Also
contains which_kernel(), which tells you which kernel in a list covers
a given time point.

ls_spice() - list all furnished spice kernels
ls_spice_pool() - list all constants in kernel constant pool
lscov() - list coverage windows of all furnished kernels of a given type
which_kernel() - which kernel(s) of a given type cover a given time point
"""

import spiceypy
import numpy as np
from collections import namedtuple
ls_spice_return=namedtuple("ls_spice_return",("type","file"))
def ls_spice(verbose=False):
    """
    List all furnished spice kernels.
    :param verbose: if True, print to stdout also
    :return: A list of ls_spice_return named tuples.
    """
    count=spiceypy.ktotal('ALL')
    result=[]
    if verbose:
        print("Total of %d kernels loaded"%count)
    for i in range(count):
        (file,type,source,handle)=spiceypy.kdata(i, 'ALL')
        if verbose:
            print("%06s %s"%(type, file))
        result.append(ls_spice_return(type=type,file=file))
    return result

def ls_spice_pool(verbose=True):
    """
    List all constants in the Spice constant pool
    :param verbose: If true, print to stdout also
    :return: Dictionary of kernel constants
    """
    kervars=spiceypy.gnpool('*',0,1000,81)
    result={}
    for kervar in sorted(kervars):
        (n,type)=spiceypy.dtpool(kervar)
        if verbose:
            print("%-50s %s %d"%(kervar,type,n))
        if type=='N':
            values=spiceypy.gdpool(kervar,0,n)
            result[kervar]=values
            if verbose:
                print(values)
        elif type=='C':
            values=spiceypy.gcpool(kervar,0,n,81)
            result[kervar]=values
            if verbose:
                print(values)
    return result

def cover(file,verbose=True):
    """
    Find coverage of an individual CK or SPK file
    :param file: Filename of file to find coverage for
    :param verbose: If true, print results to stdout
    :return: List of tuples. Each tuple has two elements,
      start and end ET of coverage.

    Note -- a CK file can conceivably have more than one object.
    This routine gets the coverage for all objects, but does
    not distinguish objects. So if the file has coverage for object
    -1000 from 12:00 to 12:01 and object -2000 from 12:00 to 12:02,
    you will get as a result:
    [(12:00,12:01),(12:00,12:02)]
    with no way to distingush which window goes with which object.
    """
    result=[]
    ids = spiceypy.cell_int(1000)
    if file.upper().endswith(".BC"):
        ids = spiceypy.ckobj(file)
        type="CK"
    else:
        ids = spiceypy.spkobj(file)
        type="SPK"
    for id in ids:
        cover = spiceypy.cell_double(10000)
        if type.upper() == 'CK':
            cover = spiceypy.ckcov(file, id, False, 'INTERVAL', 0.0, 'TDB', cover)
        else:
            cover = spiceypy.spkcov(file, id, cover)
        card = spiceypy.wncard(cover)
        for i_window in range(card):
            (left, right) = spiceypy.wnfetd(cover, i_window)
            result.append((left, right))
            if verbose:
                print("%s,%s,%d,%17.6f,%s,%17.6f,%s" % (
                type, file, id, left, spiceypy.etcal(left), right, spiceypy.etcal(right)))
    return result

def lscov(type,verbose=True):
    """
    List time coverage of all furnished kernels of a given type
    :param type: Either 'CK' or 'SPK'
    :param verbose: If True, print to stdout also
    :return: A dictionary. Key is filename, value is a list of tuples giving the start and end times in ET.
    """
    result={}
    count=spiceypy.ktotal(type)
    for i in range(count):
        (file,type,source,handle)=spiceypy.kdata(i, type)
        result[file]=cover(file,verbose=verbose)
    return result

def which_kernel(type,obj,t,flush=False):
    """
    Find out which loaded kernel provides a particular kind of data
    :param type:  one of 'LSK','FK','IK','SCLK' (collectively called text kernels)
                      or 'CK','SPK' (collectively called binary kernels)
    :param obj:   Ignored for text kernels, required for binary kernels,
                      numerical object id of spacecraft or frame to search for.
    :param t:     Ignored for text kernels, required for binary kernels,
                      Spice ET of time to search for. May be an array, see below.
    :param flush: if set, clear the cache.
    :return: path to kernel which covers this search, may be a list, see below.
    Notes
    - If looking for a text kernel, finds the last (highest priority) kernel of the given type, no matter what object it may have data for.
      Consequently, this only works well when data for only one spacecraft is loaded.
    - If looking for a binary kernel, you may pass an array of times. Return value is a list of kernels which provide data for this object
      at each time, or None if there is no kernel that covers the matching time. There is a one-to-one in-order relation between the list
      of times passed in and the list of files returned.
    - Return path is what was given in furnsh() (or in a furnished metakernel), and is either absolute or relative to what the current
      directory was at the time the kernel was furnished.
    - For performance reasons, this code keeps a cache in the form of function attribute. The cache is loaded whenever /flush is set, or
      the current cache is empty, or the requested object changes. Cache performs best therefore when you ask for coverage for the same object
      repeatedly, and switch objects a minimum number of times. You should pass /flush if you know that the kernels have changed since
      the last time you called which_kernel, because checking that the kernel list hasn't changed would take up too much time and invalidate
      a lot of the performance gain the cache provides.
    """
    utype=type.upper()
    if utype=='LSK' or utype=='FK' or utype=='IK' or utype=='SCLK':
        ftype='.T'+(utype[0:2] if len(utype)>2 else utype[0])
        count=spiceypy.ktotal('TEXT')
        for i in range(count-1,-1,-1):
            (this_file,this_type,source,handle)=spiceypy.kdata(i,'TEXT')
            #print,i,this_type,this_file,'   ',strlowcase(strmid(this_file,/rev,strlen(ftype)))
            if this_file[-len(ftype):].upper()==ftype:
                return this_file.replace('//','/')
    else:
        #Set up static variables
        if not hasattr(which_kernel,"windows"):
            which_kernel.windows={}
            which_kernel.files={}
        key=(utype,obj) #Python can index a dictionary with a tuple, so don't do things the IDL way (convert to string)
        if key in which_kernel.windows and not flush: #Check if the cache contains this object and data type
            #Retrieve the set of windows for this object and data type
            this_window=which_kernel.windows[key]
            this_files=which_kernel.files[utype]
        else:
            #Build the set of windows for this object and data type
            count=spiceypy.ktotal(utype)
            max_card=1
            #index 0 is file
            #index 1 is window in file
            #index 2 is begin (0) or end (1)
            #The effect is that this is a stack of matrices, where each plane is a file, each row in a plane is a
            #window, and each cell in the row is the beginning or end of the window. As the files are read, if a new
            #file has more windows than any previous file, exactly enough more rows are added to the bottom of all
            #the planes, and their value is set to NaN so they won't be found with np.where().
            this_window=np.zeros((count,max_card,2))*float('NaN')
            this_files=[""]*count
            for i_kernel in range(count):
                (this_file,this_type,source,handle)=spiceypy.kdata(i_kernel,utype)
                this_files[i_kernel]=this_file.replace('//','/')
                this_cover=spiceypy.cell_double(10000)
                if utype=='CK':
                    this_cover=spiceypy.ckcov(this_file, obj, False, 'INTERVAL', 0.0, 'TDB', this_cover)
                else:
                    this_cover=spiceypy.spkcov(this_file, obj, this_cover)
                card=spiceypy.wncard(this_cover)
                if card>max_card:
                    #Extend all the planes
                    this_window=np.concatenate((this_window,np.zeros((count,card-max_card,2))*float('NaN')),axis=1)
                for i_window in range(card):
                    (left,right)=spiceypy.wnfetd(this_cover,i_window)
                    this_window[i_kernel,i_window,0]=left
                    this_window[i_kernel,i_window,1]=right
            which_kernel.windows[key]=this_window
            which_kernel.files[utype]=this_files
        if t is None:
            #No time specified, return the last kernel that references this object at all
            return this_files[-1]
        try:
            n_elements_t=len(t)
        except:
            n_elements_t=1
        if n_elements_t==1:
            w = np.where(np.logical_and(t>=this_window[:, :, 0],t<=this_window[:, :, 1]))
            if len(w[0])>0:
                result=this_files[w[0][-1]]
            else:
                result=None
        else:
            result=[None]*n_elements_t
            for i_t in range(n_elements_t):
                w=np.where(np.logical_and(t[i_t]>=this_window[:,:,0],t[i_t]<=this_window[:,:,1]))
                if len(w[0])>0:
                    result[i_t]=this_files[w[0][-1]]
        return result
    raise RuntimeError('Somehow fell to the bottom of which_kernel')

def make_kernel_keyword(kernel_types,object=None,et=None):
    keyword_map={"LSK":"LS_KRN",
                 "SCLK":"SCL_KRN",
                 "FK":"F_KRN",
                 "IK":"INST_KRN",
                 "CK":"C_KRN",
                 "SPK":"SP_KRN"}
    result=[]
    for kernel_type in kernel_types:
        result.append((keyword_map[kernel_type],which_kernel(kernel_type,object,et)))
    return result

if __name__=="__main__":
    #Do whatever automatic kernel furnishing you do to get the proper list of kernels. The following
    #will almost certainly not work for you.
    import emmexipy.geometry.spice #This furnishes the kernels automatically as part of import
    ck_windows=lscov('CK',True)
    print(which_kernel("CK",-62000,[700000000,700000001,600000000]))
    pass