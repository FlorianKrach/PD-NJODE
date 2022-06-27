
#import packages
import numpy as np
import sys
import matplotlib.pyplot as plt
import time

sys.path.append("../")
#from fbm import fbm
import fbm

#import python files
try:
    from . import data_utils as data_utils
except Exception:
    import data_utils as data_utils


if __name__ == '__main__':

    #set parameters
    nb_observations = 8
    dataset = "FBM"
    hurst = 0.05


    #normally fixed parameters
    dataset_id = None
    weight = 0.5
    delta_t = 0.01 #do not change
    T = 1
    start_X = np.ones((2,1))
    drift = 2.
    volatility = 0.3

    n = T/delta_t

    times = np.linspace(0.01,1,100)
    times_nb = np.linspace(0,99,100).astype(int)
    fbm_sample = fbm.fbm(100, hurst=hurst, length=1, method="daviesharte")[1:]  #FBM sample
    if dataset == "FBM":
        fbm_sample = [x+1 for x in fbm_sample]      #start from 1
    fbm_sample = np.array(fbm_sample)


    if dataset == "FBS":
        times_mtx = np.array([(i+1)*delta_t for i in range(len(fbm_sample))])
        #print(times_mtx)
        #print(fbm_sample)
        fbm_sample = np.exp(drift*times_mtx + volatility*fbm_sample)



    #get random observations
    obs_times_nb = np.random.choice(times_nb,nb_observations,replace=False)     #randomly choose observation times for path 0
    no_obs_times_nb = np.setdiff1d(times_nb,obs_times_nb)       #remaining times are obs. times for path 1
    time_ptr = np.linspace(0,100,101).astype(int)   #at every time there is exactly 1 observation
    obs_idx = np.zeros(100).astype(int)
    obs_idx[no_obs_times_nb] = 1            # 1 if observation is for path 1, 0 if for path 0
    n_obs_ot = np.array([nb_observations, 100-nb_observations]) #only needed for compute_loss

    jump_times = times[obs_times_nb]
    print("\njump times: {}\n".format(jump_times))

    #dataset_metadata = data_utils.load_metadata(stock_model_name=dataset, time_id=dataset_id)
    dataset_metadata = {
        'drift': 2., 'volatility': 0.3, 'mean': 4,
        'speed': 2., 'correlation': 0.5, 'nb_paths': 10000, 'nb_steps': 100,
        'S0': 1, 'maturity': 1., 'dimension': 1,
        'obs_perc': 0.1,
        'scheme': 'euler', 'return_vol': False, 'v0': 1, 'hurst': 0.75, 'FBMmethod': "daviesharte", 'model_name': dataset
    }
    dataset_metadata['hurst'] = hurst
    #print(dataset_metadata)
    stockmodel = data_utils._STOCK_MODELS[dataset_metadata['model_name']](**dataset_metadata)

    print(times.shape)
    test_time = time.time()
    #calculate
    opt_loss, path_t_true, path_y_true = stockmodel.compute_cond_exp(
        times, time_ptr, fbm_sample, obs_idx,
        delta_t, T, start_X, n_obs_ot,
        return_path=True, get_loss=True, weight=weight
    )
    print("{} seconds to calculate conditional expectation".format(time.time()-test_time))
    print("In the experiment setting this would take {} hours".format((time.time()-test_time)*2000/60/60))
    print(path_y_true.shape)

    #stockmodel = data_utils._STOCK_MODELS["BlackScholes"](**dataset_metadata)
    #opt_loss_2, path_t_true_2, path_y_true_2 = stockmodel.compute_cond_exp(
    #    times, time_ptr, fbm_sample, obs_idx,
    #    delta_t, T, start_X, n_obs_ot,
    #    return_path=True, get_loss=True, weight=weight
    #)

    #print(path_t_true)
    #print(path_y_true.shape)
    #print(path_y_true)

    error = 0.0
    for t in jump_times:
        t_nb = int(round(t/delta_t))-1
        t_1 = path_t_true[2*t_nb+1]
        t_2 = path_t_true[2*t_nb+2]
        y_1 = path_y_true[2*t_nb+1,0,0]
        y_2 = path_y_true[2*t_nb+2,0,0]
        #print("jump from {} to {} at time {}(/{})".format(y_1,y_2,round(t_1,2),round(t_2,2)))
        #print("should jump to {} at time {}".format(fbm_sample[t_nb],round(times[t_nb],2)))
        error+=abs(fbm_sample[t_nb]-y_2)
    print("Jump Error= {}".format(error))

    times = np.append(0,times)
    fbm_sample = np.append(1,fbm_sample)
    plt.plot(times, fbm_sample)
    plt.plot(path_t_true, path_y_true[:, 0, :])
    #plt.plot(path_t_true_2, path_y_true_2[:, 0, :])

    plt.show()