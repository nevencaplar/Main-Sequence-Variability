"""
Created on Mon Sep  3 15:44:00 2018

@author: Neven Caplar
@contact: ncaplar@princeton.edu

Project by Sandro Tacchella (Cfa) and Neven Caplar
"""
from __future__ import division
import numpy as np
import pandas as pd
import io
from tqdm import tqdm
from scipy import interpolate


ACFData=np.loadtxt(open('./ACFTableLargeNov10.csv', "rb"), delimiter=",", skiprows=0)
ACFData[:,1]=np.round(ACFData[:,1],2)


tau=np.unique(ACFData[:,0])
slope=np.unique(ACFData[:,1])
time=np.unique(ACFData[:,2])
ACF=ACFData[:,3]


print('These auto-correlation function have been computed numerically in Wolfram Mathematica (notebook also avaliable in the Github folder) for PSD=1/(1+(f/f_bend)^(slope)),where tau=1/f_bend and f is frequency.')
print('They are tabulated as function of tau (invserse frequncy of the break in PSD), slope(high frequency slope of the PSD) and time.')
print('avaliable tau (in units of \' time units (t.u.)\')are: '+str(tau))
print('avaliable slopes are: '+str(slope))
print('largest avaliable time is [t.u.]: '+str(max(time)))

# constructing multi-index panda dataframe (series)
mi = pd.MultiIndex.from_product([tau, slope, time], names=['tau', 'slope', 'time'])

#connect multiindex to data and save as multindexed series
sr_multi = pd.Series(index=mi, data=ACF.flatten())


def get_ACF(tau,slope):
    """!gives autocorrelation function as a 2d numpy array [time, ACF]

    @param[in] tau          Decorrelation time
    @param[in] slope        high frequency slope of the PSD



    """
    
    #pull out a dataframe with tau = 100 time units (see all options above)
    select_tau=sr_multi.xs(tau, level='tau').unstack(level=0)
    #pull out a dataframe with slope = 2
    select_tau_and_slope=select_tau[slope]
    
    res=[]
    for j in range(1,len(select_tau_and_slope.values)):
        res.append([j,select_tau_and_slope.values[j]])
    
    res=np.array(res)
    return res

def get_scatter_MS(tau,slope,tMax=None,t_avg=None,convolving_array=None):
    """!gives size of scatter as a 2d numpy array [time, scatter]

    @param[in] tau          Decorellation time
    @param[in] slope        high frequency slope of the PSD
    @param[in] tmax         what is the largest time that you want to consider (see 'largest avaliable time is' above) 
                            if unsure leave empty
    @param[in] t_avg        give result at which time; if unspeciffied gives the full array for all avaliable times


    """
    if tMax is None:
        tMax=int(max(time))
    
    ACF=get_ACF(tau,slope)
    
    res=[]
    if convolving_array is None:   
        for t in range(1,tMax):
            #print(ACF[:,1][:t])
            res.append([t,(1+2*np.sum(((1-np.array(range(1,t+1))/t))*ACF[:,1][:t]))**(1/2)*(1/(t**(1/2)))])
    else:
        assert (np.sum(convolving_array) > 0.995) & (np.sum(convolving_array) < 1.005)

        t=np.min(np.array([len(ACF),len(convolving_array)]))
        
        ACF_with_0=np.vstack((np.array([0,1]),ACF))
        convolving_2d_array=np.outer(convolving_array[:t],convolving_array[:t])
        res_int=[]
        for i in range(t):
            for j in range(t):
                res_int.append(convolving_2d_array[i,j]*ACF_with_0[np.abs(i-j),1])
                
        res=np.sqrt(sum(res_int))

        return np.array([t,res])
        
    res=np.array(res)
    if t_avg is None:
        return res
    else:
        assert t_avg<tMax
        print(res)
        return res[int(t_avg-1)]  

def mean_power_10(t,x0,sigma,tau_decor):
    """! helping function that simulates averaging of the log space, assuming damped random walk 

    @param[in] tau          Decorellation time
    @param[in] slope        high frequency slope of the PSD
    @param[in] tmax         what is the largest time that you want to consider (see 'largest avaliable time is' above);


    """  
    
    
    return 10**(np.exp(-t/(2*tau_decor))*x0+sigma**2*(1-np.exp(-t/tau_decor))*np.log(10)/2)    
    
    
def get_mean_relation(tau_break,Tmax=None,sigmaMS=None):
    """!gives ratio between mean actual Delta MS and measured MS given some averaging timescale tMax
        assumes nonchanging mean sequence!

    @param[in] tau          Decorellation time
    @param[in] slope        high frequency slope of the PSD
    @param[in] tmax         what is the largest time that you want to consider (see 'largest avaliable time is' above);


    """
    
    tau_decor=tau_break/(2*np.pi*2)
    
    if Tmax is None:
        Tmax=100
        
    if sigmaMS is None:
        sigmaMS=0.4
        
    res_mean=[]
    for x0 in np.arange(-2,2.1,0.025):
        res=[]
        for t in range(Tmax):
            res.append(mean_power_10(t,x0,sigmaMS,tau_decor))
    
        res=np.array(res)
        res_mean.append([x0,np.log10(np.mean(res))])
    res_mean=np.array(res_mean)
    
    return res_mean

    
def get_mean_relation_convolution(Delta,tau,convolving_array):
    """!gives ratio between mean SFR of a longer indicator and the SFR in a shorten indicator [time, ratio of two indicators ]
        assumes nonchanging mean sequence!

    @param[in] tau          Decorellation time
    @param[in] slope        high frequency slope of the PSD
    @param[in] tmax         what is the largest time that you want to consider (see 'largest avaliable time is' above);


    """
    
    ACF=get_ACF(tau,2)
    
    tMax=np.min(np.array([len(ACF),len(convolving_array)]))
    
    #ACF=get_ACF(tau,slope)
    ACF_with_0=np.vstack((np.array([0,1]),ACF))
    

    res=np.sum(Delta*ACF_with_0[:,1][:tMax]*convolving_array[:tMax]+np.log(10)/2*(Delta**2)*ACF_with_0[:,1][:tMax]*convolving_array[:tMax])

    return res

def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample
    
def create_MS_scatter_at_given_t_interpolation(t_longer):
    """! gives interpolation of the main sequence when measured with an indicator that last for 't_longer' time units

    @param[in] t_longer         time duration of the response step function          
    
    

    """
    
    convolving_array=np.ones(t_longer)/t_longer
    
    slope_1=slope[slope>1]
    

    MS_slope_at_given_t_longer=[]
    for plot_slope in tqdm(slope_1):
        for plot_tau in tau: 
            MS_slope_at_given_t_longer.append([plot_tau,plot_slope,get_scatter_MS(plot_tau,plot_slope,None,None,convolving_array)[1]])
        
    MS_slope_at_given_t_longer_1=np.array(MS_slope_at_given_t_longer)[:,2]      
    MS_slope_at_given_t_longer_1=MS_slope_at_given_t_longer_1.reshape(len(slope_1),len(tau))
        
    MS_slope_at_given_t_longer_1_interpolation = interpolate.interp2d(tau, slope_1, MS_slope_at_given_t_longer_1, kind='cubic')  
    return MS_slope_at_given_t_longer_1_interpolation

def create_offset_slope_at_given_t_interpolation(t_longer):

    slope_1=slope[slope>1]
    

    offset_slope_at_given_t_longer=[]
    for plot_slope in tqdm(slope_1):
        for plot_tau in tau: 
            if int(t_longer)==1:
                offset_slope_at_given_t_longer.append([plot_tau,plot_slope,0])
            else:
                offset_slope_at_given_t_longer.append([plot_tau,plot_slope,get_mean_relation(plot_tau,plot_slope)[int(t_longer)-1][1]])
        
    offset_slope_at_given_t_longer_1=np.array(offset_slope_at_given_t_longer)[:,2]      
    offset_slope_at_given_t_longer_1=offset_slope_at_given_t_longer_1.reshape(len(slope_1),len(tau))
        
    offset_slope_at_given_t_longer_1_interpolation = interpolate.interp2d(tau, slope_1, offset_slope_at_given_t_longer_1, kind='cubic')  
    return offset_slope_at_given_t_longer_1_interpolation


def create_Number_of_sigmas(MS_slope_at_given_t_longer_1_interpolation,Measurment_Of_Scatter_Ratio,err_Measurment_Of_Scatter_Ratio):
    """! gives inumber of sigmas that the measurments is distance from predicted width of main sequence

    @param[in] MS_slope_at_given_t_longer_1_interpolation     fine interpolation of width as a function of parameters    
    @param[in] Measurment_Of_Scatter_Ratio     measurment of the two widths 
    @param[in] err_Measurment_Of_Scatter_Ratio     error on measurment
    
    

    """    
    
    
    slope_fine=np.arange(1.1,2.9,0.01)
    tau_fine=np.arange(1,1000,1)
    
    Number_of_sigmas_deviation_1=[]
    for plot_slope in tqdm(slope_fine):
        for plot_tau in tau_fine: 
            Number_of_sigmas_deviation_1.append(abs((MS_slope_at_given_t_longer_1_interpolation(plot_tau,plot_slope)-Measurment_Of_Scatter_Ratio)/err_Measurment_Of_Scatter_Ratio))
            
    Number_of_sigmas_deviation_1=np.array(Number_of_sigmas_deviation_1)
    Number_of_sigmas_deviation_1=Number_of_sigmas_deviation_1.reshape(len(slope_fine),len(tau_fine))

    Number_of_sigmas_deviation_reshaped_1=np.copy(Number_of_sigmas_deviation_1)
    Number_of_sigmas_deviation_reshaped_1=Number_of_sigmas_deviation_reshaped_1.reshape(len(slope_fine),len(tau_fine))  
    
    best_solution_1=[]
    for i in range(len(Number_of_sigmas_deviation_reshaped_1)):
        best_solution_1.append([slope_fine[i],tau_fine[np.argmin(Number_of_sigmas_deviation_reshaped_1[i])],Number_of_sigmas_deviation_reshaped_1[i][np.argmin(Number_of_sigmas_deviation_reshaped_1[i])]])
        
    best_solution_1=np.array(best_solution_1)
    best_solution_1=best_solution_1[best_solution_1[:,2]<0.1]
    
    return tau_fine,slope_fine,Number_of_sigmas_deviation_reshaped_1,best_solution_1
    
def create_Number_of_sigmas_offset(offset_slope_at_given_t_longer_1_interpolation,Measurment_Of_Offset_Slope,err_Measurment_Of_Offset_Slope):
    
    slope_fine=np.arange(1.1,2.9,0.01)
    tau_fine=np.arange(1,200,0.1)
    
    Number_of_sigmas_deviation_1=[]
    for plot_slope in tqdm(slope_fine):
        for plot_tau in tau_fine: 
            Number_of_sigmas_deviation_1.append(abs((offset_slope_at_given_t_longer_1_interpolation(plot_tau,plot_slope)-Measurment_Of_Offset_Slope)/err_Measurment_Of_Offset_Slope))
            
    Number_of_sigmas_deviation_1=np.array(Number_of_sigmas_deviation_1)
    Number_of_sigmas_deviation_1=Number_of_sigmas_deviation_1.reshape(len(slope_fine),len(tau_fine))

    Number_of_sigmas_deviation_reshaped_1=np.copy(Number_of_sigmas_deviation_1)
    Number_of_sigmas_deviation_reshaped_1=Number_of_sigmas_deviation_reshaped_1.reshape(len(slope_fine),len(tau_fine))  
    
    best_solution_1=[]
    for i in range(len(Number_of_sigmas_deviation_reshaped_1)):
        best_solution_1.append([slope_fine[i],tau_fine[np.argmin(Number_of_sigmas_deviation_reshaped_1[i])],Number_of_sigmas_deviation_reshaped_1[i][np.argmin(Number_of_sigmas_deviation_reshaped_1[i])]])
        
    best_solution_1=np.array(best_solution_1)
    best_solution_1=best_solution_1[best_solution_1[:,2]<0.1]
    
    return tau_fine,slope_fine,Number_of_sigmas_deviation_reshaped_1,best_solution_1

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

def create_Number_of_R_interpolation(convolving_array):
    Number_of_R=[]
    slope_1=slope[slope>1]
    for plot_slope in tqdm(slope_1):
        for plot_tau in tau: 
            single_ACF=get_ACF(plot_tau,plot_slope)[:,1]

            t=np.min(np.array([len(single_ACF),len(convolving_array)]))
            sum_ACF_over_inf_response_function=np.sum(single_ACF[:t]*convolving_array[:t])
    
            sum_single_ACF_over_R=[]
            for i in range(1,len(single_ACF)):
                sum_single_ACF_over_R.append(np.sum(single_ACF[:i])/i)
    
            sum_single_ACF_over_R=np.array(sum_single_ACF_over_R)
            Number_of_R.append(find_nearest(sum_single_ACF_over_R,sum_ACF_over_inf_response_function)[1])
    Number_of_R=np.array(Number_of_R)
    Number_of_R=Number_of_R.reshape(19,29)
    Number_of_R_interpolation = interpolate.interp2d(tau,slope_1, Number_of_R, kind='linear')    
    return Number_of_R_interpolation

def create_Number_of_R_interpolation_Variance(convolving_array):
    Number_of_R=[]
    slope_1=slope[slope>1]
    convolving_array=convolving_array[np.cumsum(convolving_array)<0.998]
    for plot_slope in tqdm(slope_1):
        for plot_tau in tau:
            ACF=get_ACF(plot_tau,plot_slope)

            t=np.min(np.array([len(ACF),len(convolving_array)]))
            convolving_2d_array=np.outer(convolving_array[:t],convolving_array[:t])   
            ACF_with_0=np.vstack((np.array([0,1]),ACF))            
            
            res_int=convolving_2d_array* np.fromfunction(lambda i, j: ACF_with_0[np.abs(i-j),1], (t, t),dtype=int)
            """
            old code, removed on Oct12, 2018
            res_int=[]
            for i in range(t):
                for j in range(t):
                    res_int.append(convolving_2d_array[i,j]*ACF_with_0[np.abs(i-j),1])
            """            
            res=[]
            for tv in range(1,t):
                res.append([tv,(1/(tv))*(1+2*np.sum(((1-np.array(range(1,tv+1))/tv))*ACF[:,1][:tv]))])
                
            res=np.array(res) 
            Number_of_R.append(list(np.abs(res[:,1]-np.sum(res_int))).index(np.min(np.abs(res[:,1]-np.sum(res_int))))+1)


    Number_of_R=np.array(Number_of_R)
    Number_of_R=Number_of_R.reshape(19,29)
    Number_of_R_interpolation = interpolate.interp2d(tau,slope_1, Number_of_R, kind='linear')    
    return Number_of_R_interpolation


def create_Number_of_R_parameter_space(Number_of_R_interpolation):
    
    slope_fine=np.arange(1.1,2.9,0.01)
    tau_fine=np.arange(1,200,0.1)
    
    Number_of_R_parameter_space=[]
    for plot_slope in tqdm(slope_fine):
        for plot_tau in tau_fine: 
            Number_of_R_parameter_space.append(int(Number_of_R_interpolation(plot_tau,plot_slope)))
    
    
    Number_of_R_parameter_space=np.array(Number_of_R_parameter_space)
    
    return slope_fine,tau_fine,Number_of_R_parameter_space

def ConnectionBetweenLinearAndLogVariance(log_var):
    x=log_var
    return 10**(-3.42755 + 39.9722*x - 248.007*x**2 + 834.61*x**3 - 1329.93*x**4 + 807.497*x**5)