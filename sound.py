import wave
import numpy as np
import matplotlib.pyplot as plt

def linterp(x0,y0,x1,y1,x):
    t=(x-x0)/(x1-x0)
    return (1-t)*y0+t*y1

def cyclesound(A,hz,t0,t1,samplerate,oufn=None):
    """

    :param A: callable (I suggest interp1d output) providing amplitude at given time, varying from 0 to 32767=full-scale
    :param hz: callable providing frequency at given time in Hz
    :param t0: Initial time
    :param t1: final time
    :param samplerate: final sound file sample rate in Hz -- 44100 matches a CD, 48000 matches a DVD
    :param oufn: if present, write data to a wav file as well
    :return: Numpy array of data
    """
    #This code painstakingly constructs the whole sound cycle by cycle. For each cycle, it:
    #  * Figures out exactly where it starts (in continuous time)
    #  * Figures out exactly how long the cycle is
    #  * Figures out all the samples covered by this time
    #  * Samples a one-cycle perfect sine wave at these points, scaled by amplitude
    #  * Writes the samples to the output in their proper place
    tsamples=np.arange(t0,t1,1.0/samplerate)
    samples=np.zeros(tsamples.size,dtype=np.int16)
    t=t0
    markpoint=0
    dmark=1
    while t<t1:
        tp=t+1.0/hz(t) #tp is the end of the current cycle
        w=np.where(np.logical_and(tsamples>=t,tsamples<tp))[0]
        arg=linterp(t,0,tp,np.pi*2,tsamples[w])
        samples[w]=A(t)*np.sin(arg)
        t=tp
        if t>markpoint:
            print(markpoint,t1)
            markpoint+=dmark
    if oufn is not None:
        with wave.open(oufn, "wb") as ouf:
            ouf.setnchannels(1)
            ouf.setframerate(samplerate)
            ouf.setsampwidth(2)
            ouf.writeframes(samples)
    return samples

def framesound(A,hz,t0,t1,samplerate,framerate=24.0,oufn=None):
    """

    :param A: callable (I suggest interp1d output) providing amplitude at given time, varying from 0 to 32767=full-scale
    :param hz: callable providing frequency at given time in Hz
    :param t0: Initial time
    :param t1: final time
    :param samplerate: final sound file sample rate in Hz -- 44100 matches a CD, 48000 matches a DVD
    :param framerate: Video frame rate in Hz -- see below why this is used
    :param oufn: if present, write data to a wav file as well
    :return: Numpy array of data
    """
    #This code constructs the sound frame by frame. During a frame, the sound has constant
    #amplitude and wavelength. For each frame:
    #   * Figure out the current frequency and amplitude at the beginning of the current frame
    #   * Figure out the end of cycle nearest to the next frame
    #   * Evaluate the wave over the whole frame
    #   * Write the frame data to the wave
    samples=np.zeros(int((t1-t0)*samplerate),dtype=np.int16)
    t=t0
    i_frame=0
    i_sample=0
    while t<t1:
        #   * Figure out the current frequency at the beginning of the current frame
        cycle_T=1.0/hz(t) #Length of one cycle in this frame
        #   * Figure out the end of cycle nearest to the next frame
        ideal_tp=(i_frame+1)/framerate #ideal end of current frame, exactly on a frame boundary
        cycle_time=ideal_tp-t #Exact amount of time in this frame, t might not be exactly on a frame boundary, but tp will be
        ideal_cycles_in_frame=cycle_time/cycle_T #exact number of cycles in this frame
        cycles_in_frame=int(ideal_cycles_in_frame+0.5)
        tp=t+cycles_in_frame*cycle_T
        ideal_i_samplep=tp*samplerate
        i_samplep=int(ideal_i_samplep+0.5)
        if i_samplep>len(samples):
            i_samplep=len(samples)
        #   * Evaluate the wave over the whole frame
        arg=linterp(i_sample,0,i_samplep,2*np.pi*cycles_in_frame,np.arange(i_sample,i_samplep))
        samples[i_sample:i_samplep]=A(t)*np.sin(arg)
        t=tp
        i_frame+=1
        i_sample=i_samplep

        #   * Write the frame data to the wave
    if oufn is not None:
        with wave.open(oufn, "wb") as ouf:
            ouf.setnchannels(1)
            ouf.setframerate(samplerate)
            ouf.setsampwidth(2)
            ouf.writeframes(samples)
    return samples


sound=framesound

if __name__=="__main__":
    sound(lambda x:32767-x*1000,lambda x:440+x*44,0,10,48000,"chirp_10.wav")
