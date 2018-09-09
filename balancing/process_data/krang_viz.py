import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab



# Read Data
runs = [str(x) for x in [0,1,2]]
betas = [str(x) for x in [16,32,64,190]]

full = {}
for beta in betas:
    full[beta] = {}
    for run in runs:
        full[beta][run] = {}

        data_file = "beta"+str(beta)+"_"+str(run)

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

        start = np.argmax(np.abs(data[:,1])>1)
        end = np.argmax(time-time[start]>60)

        steady = end-np.argmax(np.abs(data[end:start:-1,1])>0.2*np.max(np.abs(data[start:end,1])))

        full[beta][run]['trim'] = data[start:end,:]
        full[beta][run]['steady'] = data[steady:end,:]
        full[beta][run]['time'] = time[start:end]-time[start]


for beta in betas:
    full[beta]['all'] = {}
    full[beta]['all']['trim'] = full[beta][runs[0]]['trim']
    for run in runs[1:]:
        full[beta]['all']['trim'] = np.append(full[beta]['all']['trim'],full[beta][run]['trim'],axis=0)


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
    plt.subplot(4,2,2*i+2)    
    x = np.abs(full[beta]['0']['steady'][:,1])
    (mu,sigma) = norm.fit(x)
    print mu,sigma
    n, bins, patches = plt.hist(x,bins=np.linspace(0,10,100),normed=1)
    y = mlab.normpdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)

    
    plt.subplot(4,2,2*i+1)
    for run in runs:
        t = full[beta][run]['time']
        x = np.convolve(full[beta][run]['trim'][:,1],np.ones(10)/10.0)
        plt.plot(t,x[:len(t)],label="run "+run)
        
        # x = np.convolve(full[beta][run]['steady'][:,1],np.ones(10)/10.0)
        # plt.plot(x,label="run "+run)
    plt.title(beta)
    plt.ylim(-40,20)
    plt.legend()

    # plt.plot(np.convolve(full[beta][run]['trim'][:,1],np.ones(10)/10.0))



for beta in betas:
    plt.figure()
    for run in runs:
        t = full[beta][run]['time']
        x = np.convolve(full[beta][run]['trim'][:,1],np.ones(10)/10.0)
        plt.plot(t,x[:len(t)],label="run "+run)
    plt.title(beta)
    plt.ylim(-40,20)
    plt.legend()



plt.show()

