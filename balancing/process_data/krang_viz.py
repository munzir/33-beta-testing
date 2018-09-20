import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Read Data
runs = [str(x) for x in [0,1,2,3,4,5,6]]
betas = [str(x) for x in [16,32,64,128,190]]

full = {}
for beta in betas:
    full[beta] = {}
    for run in runs:
        full[beta][run] = {}

        data_file = "sergio_data/betaVectors1initialBetahardwareAll190Vectorsnum"+str(beta)+"trial"+str(run)+"statedump.txt"
        data_file = "sergio_data/betaVectors1initialBetahardwareAll190Vectorsnum"+str(beta)+"trial"+str(run)+"statedump.txt"

        with open(data_file) as f:
            lines = f.readlines()

        raw_data = [x.strip() for x in lines]

        rows = len(raw_data)
        cols = len(raw_data[0].split())

        data = np.zeros((rows,cols))

        for i,row_space in enumerate(raw_data):
            row = row_space.split()
            data[i,:] =  row

        time = np.cumsum(data[:,0])


        full[beta][run] = {}
        full[beta][run]['raw'] = data
        full[beta][run]['raw_time'] = time

        N = 20.0
        data[:,1] = np.convolve(data[:,1],np.ones(int(N))/N)[int(N/2):-int(N/2)+1]
        # print smoothed.shape, data[:,1].shape
        # exit()

        start = np.argmax(np.abs(data[:,1])>1)
        end = np.argmax(time-time[start]>55)
        t20 = np.argmax(time-time[start]>30)

        # steady = end-np.argmax(np.abs(data[end:start:-1,1])>0.4*np.max(np.abs(data[start:end,1])))

        steady = t20-np.argmax(np.abs(data[t20:start:-1,1])>0.4*np.max(np.abs(data[start:t20,1])))

        # pos = np.mean(data[t20:end,5])
        # norm_pos = data[:,5]-pos
        # steady_pos = end-np.argmax((np.abs(norm_pos[end:start:-1])>0.05*np.max(np.abs(norm_pos[start:end]))))


        # GET AVERAGE POSITION

        # plt.figure()
        # # plt.plot(20*(np.abs(data[t20:start:-1,1])>0.25*np.max(np.abs(data[start:t20,1])))[::-1])
        # plt.plot((np.abs(norm_pos[end:start:-1])>0.05*np.max(np.abs(norm_pos[start:end])))[::-1])
        # plt.plot(norm_pos[start:end])
        
        # plt.figure()
        # plt.plot(data[start:steady_pos,1])
        # plt.plot(data[start:steady_pos,5])
        # plt.show()
        
        
        full[beta][run]['trim'] = data[start:end,:]
        full[beta][run]['time'] = time[start:end]-time[start]


        t10 = np.argmax(full[beta][run]['time']>10)
        t20 = np.argmax(full[beta][run]['time']>20)
        t40 = np.argmax(full[beta][run]['time']>10)

        
        pos = np.mean(full[beta][run]['trim'][t20:,5])
        norm_pos = full[beta][run]['trim'][:,5]-pos
        steady_pos = len(norm_pos)-np.argmax((np.abs(norm_pos[-1::-1])>0.3))#0.04*np.max(np.abs(norm_pos))))


        # Torque = Current*15*km = Current*1.2708
        # Transform current to torque
        km = 12*0.00706  
        full[beta][run]['power'] = np.abs(np.multiply(full[beta][run]['trim'][:,1]*15*km  , full[beta][run]['trim'][:,6]))
        full[beta][run]['energy'] = np.cumsum(full[beta][run]['power'])
        full[beta][run]['current'] = full[beta][run]['power'] #full[beta][run]['trim'][:,1]
        # full[beta][run]['current'] = full[beta][run]['trim'][:,1]

        
        full[beta][run]['trans'] = full[beta][run]['power'][:t10]
        full[beta][run]['steady'] = full[beta][run]['power'][t10:]


        # N = 20.0
        # smoothed = np.convolve(full[beta][run]['trim'][:,1],np.ones(int(N))/N)

        integral = np.abs(np.multiply(full[beta][run]['trim'][:,1],data[start:end,0]))
        trans_integral = np.sum(integral[:t20])
        steady_integral = np.sum(integral[t20:])


        # full[beta][run]['peak_power'] = np.max(full[beta][run]['power'])
        # full[beta][run]['steady_power'] = np.mean(full[beta][run]['power'][])



        full[beta][run]['overshoot_pos'] = 0.27*np.max(full[beta][run]['trim'][:,1])
        full[beta][run]['steady_pos_time'] = time[steady_pos]
        full[beta][run]['steady_pos'] = 0.27*pos

        full[beta][run]['overshoot_power'] = np.max(full[beta][run]['trans'])   
        full[beta][run]['steady_avg_power'] = np.mean(np.abs(np.multiply(full[beta][run]['steady'],full[beta][run]['trim'][t10:,0])))

        full[beta][run]['trans_integral'] = (1.0/3600)*trans_integral
        full[beta][run]['steady_integral'] = (1.0/3600)*steady_integral

        # full[beta][run]['resting_pos'] = full[beta][run]['trim'][-1,5]
        # full[beta][run]['overshoot_pos'] = np.argmax(np.abs(full[beta][run]['trim'][:,5]))
        
        

        


test = {}
for beta in betas:
    print
    test[beta] = {}
    for value in ['overshoot_pos','steady_pos','steady_pos_time','overshoot_power','steady_avg_power']:
        avg = np.average([full[beta][run][value] for run in runs])
        std = np.std([full[beta][run][value] for run in runs])

        test[beta][value] = avg

        # x = "{0:.3g}".format(avg)
        # d = len(x.split(".")[1])

        # print beta, value, "{0:.3g} \\pm".format(avg), round(std,d)



        print beta, value, avg,std
        # plt.figure()
        # # plt.plot(20*(np.abs(data[t20:start:-1,1])>0.25*np.max(np.abs(data[start:t20,1])))[::-1])
        # plt.plot((np.abs(norm_pos[end:start:-1])>0.05*np.max(np.abs(norm_pos[start:end])))[::-1])
        # plt.plot(norm_pos[start:end])
        
        # plt.figure()
        # plt.plot(data[start:steady_pos,1])
        # plt.plot(data[start:steady_pos,5])
        # plt.show()

for beta in betas:
    for value in ['overshoot_pos','steady_pos','steady_pos_time','overshoot_power','steady_avg_power']:
        avg = np.average([full[beta][run][value] for run in runs])
        std = np.std([full[beta][run][value] for run in runs])

        print value, 1.0*(test[betas[0]][value]-avg)/test[betas[0]][value]



# exit()
# plt.figure()
# colors = ['r','k','b','k','g']
# for b,beta in enumerate(betas):
#     full[beta]['all'] = {}
#     full[beta]['all']['trim'] = full[beta][runs[0]]['trim']

#     full[beta]['all']['current'] = {}
#     array_len = min([len(full[beta][run]['current']) for run in runs])
#     full[beta]['all']['current']['stack'] = np.zeros((array_len,len(runs)))
#     for i,run in enumerate(runs[1:]):
#         print i
#         full[beta]['all']['trim'] = np.append(full[beta]['all']['trim'],full[beta][run]['trim'],axis=0)

#         full[beta]['all']['current']['stack'][:,i] = full[beta][run]['current'][:array_len]

#     full[beta]['all']['current']['mean'] = np.mean(full[beta]['all']['current']['stack'],axis=1)
#     full[beta]['all']['current']['std'] = np.std(full[beta]['all']['current']['stack'],axis=1)

#     t = full[beta][runs[0]]['time'][:array_len]
#     mean = full[beta]['all']['current']['mean']
#     std = full[beta]['all']['current']['std']

#     mean = np.convolve(mean,np.ones(1)/1.0)[:len(t)]
    
#     # t = t[::10]
#     # mean = mean[::10]
#     # std = std[::10]

#     if beta in ['32','128']:
#         continue
#     plt.plot(t,mean, label=r"$\beta_{"+str(beta)+"}$",color=colors[b],alpha=0.8)
#     # plt.plot(t,mean-1*std,alpha=0.2)#,color=colors[b])
#     # plt.plot(t,mean+1*std,alpha=0.2)#,color=colors[b])
#     # plt.fill_between(t,mean-0*std,mean+1*std,alpha=0.1)#,color=colors[b])
#     # plt.yscale('log')

# plt.xlabel("Time (s)")
# plt.ylabel("Log Power (log W)")
# plt.legend()
# plt.show()



# f, ax = plt.subplots(5,1,sharex=True)
# for i,beta in enumerate(betas):
#     # plt.subplot(5,1,i+1)
#     for run in runs:
#         t = full[beta][run]['time']
#         t20 = np.argmax(t>20)
#         # t = t[:t20]
#         # x = np.convolve(full[beta][run]['power'][:t20],np.ones(10)/10.0)
#         steady = t20-np.argmax(np.abs(full[beta][run]['power'][t20:start:-1])>0.4*np.max(np.abs(full[beta][run]['power'][start:t20])))
#         t = t[steady:]
#         x = full[beta][run]['energy'][steady:]-full[beta][run]['energy'][steady]
#         # x = full[beta][run]['trim'][:,5]
#         ax[i].plot(t,x[:len(t)],label="run "+run)
#     # plt.title("Beta Iteration "+beta,fontsize=18)
#     # plt.ylim(-40,20)
#     # ax[i].set_ylim(-10,150)




f, ax = plt.subplots(5,2,sharex='col')
colors = ['r','k','b','k','g']
for j in [0,1]:
    for b,beta in enumerate(betas):
        full[beta]['all'] = {}
        full[beta]['all']['trim'] = full[beta][runs[0]]['trim']

        full[beta]['all']['current'] = {}
        array_len = min([len(full[beta][run]['current']) for run in runs])
        t10 = np.argmax(full[beta][run[0]]['time']>10)

        print array_len,

        if j==0:
            s = t10
        else:
            s = array_len-t10


        print array_len,t10

        full[beta]['all']['current']['stack'] = np.zeros((s,len(runs)))
        for i,run in enumerate(runs[1:]):
            print i
            full[beta]['all']['trim'] = np.append(full[beta]['all']['trim'],full[beta][run]['trim'],axis=0)

            if j==0:
                full[beta]['all']['current']['stack'][:,i] = full[beta][run]['current'][:s]
            else:
                full[beta]['all']['current']['stack'][:,i] = full[beta][run]['current'][t10:array_len]

        full[beta]['all']['current']['mean'] = np.mean(full[beta]['all']['current']['stack'],axis=1)
        full[beta]['all']['current']['std'] = np.std(full[beta]['all']['current']['stack'],axis=1)

        if j==0:
            t = full[beta][runs[0]]['time'][:s]
        else:
            t = full[beta][runs[0]]['time'][t10:array_len]
        mean = full[beta]['all']['current']['mean']
        std = full[beta]['all']['current']['std']

        mean = np.convolve(mean,np.ones(1)/1.0)[:len(t)]
        
        # t = t[::10]
        # mean = mean[::10]
        # std = std[::10]

        # if beta in ['32','128']:
        #     continue
        # ax[b].plot(t,mean, label=r"$\beta_{"+str(beta)+"}$",color='r',alpha=1.0)
        # ax[b].plot(t,mean-1*std,alpha=0.2,color=colors[b])
        # ax[b].plot(t,mean+1*std,alpha=0.2,color=colors[b])
        
        # ax[b,j].fill_between(t,mean-1.5*std,mean+1.5*std,alpha=1.0,color='lightgrey')

        for i,run in enumerate(runs[1:]):
            ax[b,j].plot(t,full[beta]['all']['current']['stack'][:,i],alpha=0.8)#,color='g')

        # ax[b,j].plot(t,mean, label=r"$\beta_{"+str(beta)+"}$",color='k',alpha=1.0)

        # plt.yscale('log')

        ax[b,j].text(0.95,0.75,r"$\beta_{"+str(beta)+"}$", verticalalignment='top', horizontalalignment='right',transform=ax[b,j].transAxes,fontsize=18, bbox={'facecolor':'white', 'alpha':1.0, 'pad':10})

        if b == len(betas)-1:
            ax[b,j].set_xlabel(r'Time (s)',fontsize=18)
        if b == len(betas)/2:
            if j == 0:
                ax[b,j].set_ylabel(r'Instantaneous Power during Standing Action (W)',fontsize=18)
            else:
                ax[b,j].set_ylabel(r'Instantaneous Power during Balancing Action (W)',fontsize=18)

        if j==0:
            ax[b,j].set_ylim(-10,160)
        else:
            ax[b,j].set_ylim(-1,13)
            ax[b,j].set_xticks(np.arange(10,60,10))

        ax[b,j].tick_params(labelsize=18)
        ax[b,j].grid()

    # plt.xlabel("Time (s)")
    # plt.ylabel("Log Power (log W)")
    # plt.legend()
plt.show()







# f, ax = plt.subplots(5,1,sharex=True)
# colors = ['r','k','b','k','g']
# for b,beta in enumerate(betas):
#     full[beta]['all'] = {}
#     full[beta]['all']['trim'] = full[beta][runs[0]]['trim']

#     full[beta]['all']['current'] = {}
#     array_len = min([len(full[beta][run]['current']) for run in runs])
#     full[beta]['all']['current']['stack'] = np.zeros((array_len,len(runs)))
#     for i,run in enumerate(runs[1:]):
#         print i
#         full[beta]['all']['trim'] = np.append(full[beta]['all']['trim'],full[beta][run]['trim'],axis=0)

#         full[beta]['all']['current']['stack'][:,i] = full[beta][run]['current'][:array_len]

#     full[beta]['all']['current']['mean'] = np.mean(full[beta]['all']['current']['stack'],axis=1)
#     full[beta]['all']['current']['std'] = np.std(full[beta]['all']['current']['stack'],axis=1)

#     t = full[beta][runs[0]]['time'][:array_len]
#     mean = full[beta]['all']['current']['mean']
#     std = full[beta]['all']['current']['std']

#     mean = np.convolve(mean,np.ones(1)/1.0)[:len(t)]
    
#     # t = t[::10]
#     # mean = mean[::10]
#     # std = std[::10]

#     # if beta in ['32','128']:
#     #     continue
#     # ax[b].plot(t,mean, label=r"$\beta_{"+str(beta)+"}$",color='r',alpha=1.0)
#     # ax[b].plot(t,mean-1*std,alpha=0.2,color=colors[b])
#     # ax[b].plot(t,mean+1*std,alpha=0.2,color=colors[b])
#     ax[b].fill_between(t,mean-1.5*std,mean+1.5*std,alpha=1.0,color='lightgrey')

#     for i,run in enumerate(runs[1:]):
#         ax[b].plot(t,full[beta]['all']['current']['stack'][:,i],alpha=0.8)#,color='g')

#     ax[b].plot(t,mean, label=r"$\beta_{"+str(beta)+"}$",color='k',alpha=1.0)

#     # plt.yscale('log')

#     ax[b].text(0.95,0.32,r"$\beta_{"+str(beta)+"}$", verticalalignment='top', horizontalalignment='right',transform=ax[b].transAxes,fontsize=18, bbox={'facecolor':'white', 'alpha':1.0, 'pad':10})

#     if b == len(betas)-1:
#         ax[b].set_xlabel(r'Time (s)',fontsize=18)
#     if b == len(betas)/2:
#         ax[b].set_ylabel(r'Current (A)',fontsize=18)

#     ax[b].tick_params(labelsize=18)
#     ax[b].grid()

# # plt.xlabel("Time (s)")
# # plt.ylabel("Log Power (log W)")
# # plt.legend()
# plt.show()







print full[beta]['all']['current']['mean'].shape
exit()
# full[beta]['all']['current']['std'] = 



plt.figure()
run = 'all'
exemplar = [0,0,0,0,0]
for i,beta in enumerate(betas):
    run = str(exemplar[i])
    t10 = np.argmax(full[beta][run]['time']>10)
    t = full[beta][run]['time'][:t10]
    # x = np.convolve(full[beta][run]['trim'][:,1],np.ones(10)/10.0)
    x = full[beta][run]['current'][:t10]
    # x = full[beta][run]['trim'][:,5]
    plt.plot(t,x[:len(t)],label="beta "+beta)
plt.legend()



titles = ["dt","leftWheel","rightWheel","theta","dtheta","x/R","dx/R","psi","dpsi"]


# pose_out_file << dt << " " << input[0] << " " << input[1] << " ";
# pose_out_file << state.transpose() << " ";
# pose_out_file << dartToMunzir(robot_->getPositions().transpose(), robot_).transpose();
# pose_out_file << endl;

# for data_file in data_files:
#     plt.figure()
#     plt.suptitle(data_file)
#     for i,title in enumerate(titles):
#         plt.subplot(round(len(titles)/2.0),2,i+1)
#         plt.plot(full[beta][run]['time'],full[beta][run]['trim'][:,i])
#         plt.title(title)

# plt.figure()
# for data_file in data_files:
#     plt.plot(full[beta][run]['time'],full[beta][run]['trim'][:,1],label=data_file)
# plt.legend()


# plt.figure()
# for data_file in data_files:
#     plt.plot(np.convolve(full[beta][run]['trim'][:,1],np.ones(10)/10.0),label=data_file)
# plt.legend()


# plt.figure()
# for data_file in data_files:
#     plt.plot(full[beta][run]['time'],np.cumsum(np.abs(full[beta][run]['trim'][:,1])))


plt.figure()
run = 'all'
for i,beta in enumerate(betas):
    plt.subplot(5,4,4*i+3)    
    x = np.abs(full[beta]['0']['steady'][:])
    (mu,sigma) = norm.fit(x)
    print mu,sigma
    n, bins, patches = plt.hist(x,bins=np.linspace(0,1,100),normed=1)
    y = mlab.normpdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)

    
    for run in runs:
        # t = full[beta][run]['time']
        # x = np.convolve(full[beta][run]['trim'][:,1],np.ones(10)/10.0)
        # plt.plot(t,x[:len(t)],label="run "+run)
        plt.subplot(5,4,4*i+1)
        x = np.convolve(full[beta][run]['trans'][:],np.ones(10)/10.0)
        plt.plot(x,label="run "+run)

        plt.subplot(5,4,4*i+2)
        t = full[beta][run]['time']
        # x = np.convolve(full[beta][run]['trim'][:,1],np.ones(10)/10.0)
        x = full[beta][run]['trim'][:,1]
        # x = full[beta][run]['trim'][:,5]
        plt.plot(t,x[:len(t)],label="run "+run)



        plt.subplot(5,4,4*i+4)
        t = full[beta][run]['time']
        x = full[beta][run]['trim'][:,5]
        plt.plot(t,x[:len(t)],label="run "+run)


    plt.title(beta)


    plt.ylim(-40,20)
    # plt.legend()

    # plt.plot(np.convolve(full[beta][run]['trim'][:,1],np.ones(10)/10.0))



f, ax = plt.subplots(5,1,sharex=True)
for i,beta in enumerate(betas):
    # plt.subplot(5,1,i+1)
    for run in runs:
        t = full[beta][run]['time']
        t20 = np.argmax(t>20)
        # t = t[:t20]
        # x = np.convolve(full[beta][run]['power'][:t20],np.ones(10)/10.0)
        steady = t20-np.argmax(np.abs(full[beta][run]['power'][t20:start:-1])>0.4*np.max(np.abs(full[beta][run]['power'][start:t20])))
        t = t[steady:]
        x = full[beta][run]['energy'][steady:]-full[beta][run]['energy'][steady]
        # x = full[beta][run]['trim'][:,5]
        ax[i].plot(t,x[:len(t)],label="run "+run)
    # plt.title("Beta Iteration "+beta,fontsize=18)
    ax[i].text(0.95,0.8,r"$\beta_{"+str(beta)+"}$", verticalalignment='top', horizontalalignment='right',transform=ax[i].transAxes,fontsize=18, bbox={'facecolor':'white', 'alpha':1.0, 'pad':10})
    # plt.ylim(-40,20)
    # ax[i].set_ylim(-10,150)

    if i == len(betas)-1:
        ax[i].set_xlabel(r'Time (s)',fontsize=18)
    if i == len(betas)/2:
        ax[i].set_ylabel(r'Power (W)',fontsize=18)

    ax[i].tick_params(labelsize=18)
    ax[i].grid()

    # plt.legend()


# plt.figure()
# for i,beta in enumerate(betas):
#     plt.subplot(5,1,i+1)
#     for run in runs:
#         t = full[beta][run]['time']
#         t20 = np.argmax(t>20)
#         t = t[:t20]
#         x = np.convolve(full[beta][run]['power'][:t20],np.ones(10)/10.0)
#         # x = full[beta][run]['trim'][:,5]
#         plt.plot(t,x[:len(t)],label="run "+run)
#     # plt.title("Beta Iteration "+beta,fontsize=18)
#     plt.text(0.95,0.95,"Beta Iteration "+beta,fontsize=18, bbox={'facecolor':'white', 'alpha':1.0, 'pad':10})
#     # plt.ylim(-40,20)
#     plt.ylim(-10,150)

#     if i == len(betas)-1:
#         plt.xlabel('Time (s)',fontsize=18)
#     if i == len(betas)/2:
#         plt.ylabel('Current (A)',fontsize=18)

#     plt.tick_params(labelsize=18)
#     plt.grid()

#     # plt.legend()

#10 10 95 95 20 60


plt.show()

