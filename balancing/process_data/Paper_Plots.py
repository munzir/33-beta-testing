import matplotlib.pyplot as plt
import numpy as np


# Read Data
data_files = ["beta"+str(i)+"_0" for i in [190,64,32,16]]

full = {}
for data_file in data_files:

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


    full[data_file] = {}
    full[data_file]['raw'] = data
    full[data_file]['raw_time'] = time
    
    start = np.argmax(np.abs(data[:,1])>1)
    end = np.argmax(time-time[start]>60)


    full[data_file]['trim'] = data[start-10:end,:]
    full[data_file]['time'] = time[start-10:end]-time[start]
    full[data_file]['dt'] = data[start-10:end,0]


titles = ["dt","leftWheel","rightWheel","theta","dtheta","x/R","dx/R","psi","dpsi"]


for data_file in data_files:
    
    print data_file,':'
    last = len(full[data_file]['trim'][:,1])-1
    max_val = np.max(np.abs(full[data_file]['trim'][:,1]))    
    max_ind = np.argmax(full[data_file]['trim'][:,1])
    mean_max = np.mean(np.abs(full[data_file]['trim'][max_ind-5:max_ind + 5,1]))
    print 'Mean maximum value: ', mean_max
    Global_max_val = 20
    mean_val = np.mean(full[data_file]['trim'][last-200:last,1])
    min_line = np.ones((last+1,1))*(mean_val-Global_max_val*0.1)
    max_line = np.ones((last+1,1))*(mean_val+Global_max_val*0.1)
    
#    
    plt.figure()
    plt.grid()
    plt.plot(full[data_file]['time'],full[data_file]['trim'][:,1])
    plt.plot(full[data_file]['time'],min_line)
    plt.plot(full[data_file]['time'],max_line)
    
    print 'Ampere*hour start: ', np.mean(full[data_file]['dt'][0:400]*np.abs(full[data_file]['trim'][0:400,1]))
    print 'Ampere*hour end: ', np.mean(full[data_file]['dt'][1500:1700]*np.abs(full[data_file]['trim'][1500:1700,1]))


    
    

    
    
    











