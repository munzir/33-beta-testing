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

    full[data_file]['trim'] = data[start:end,:]
    full[data_file]['time'] = time[start:end]-time[start]



titles = ["dt","leftWheel","rightWheel","theta","dtheta","x/R","dx/R","psi","dpsi"]

# pose_out_file << dt << " " << input[0] << " " << input[1] << " ";
# pose_out_file << state.transpose() << " ";
# pose_out_file << dartToMunzir(robot_->getPositions().transpose(), robot_).transpose();
# pose_out_file << endl;

for data_file in data_files:
    plt.figure()
    plt.suptitle(data_file)
    for i,title in enumerate(titles):
        plt.subplot(round(len(titles)/2.0),2,i+1)
        plt.plot(full[data_file]['time'],full[data_file]['trim'][:,i])
        plt.title(title)

plt.figure()
for data_file in data_files:
    plt.plot(full[data_file]['time'],full[data_file]['trim'][:,1],label=data_file)
plt.legend()


plt.figure()
for data_file in data_files:
    plt.plot(full[data_file]['time'],np.cumsum(np.abs(full[data_file]['trim'][:,1])))



plt.show()

# print data
# print rows, cols





# print line

# print raw_data

#     for xx in x:
#         content.append(xx)

# content = [x.strip() for x in content]

# print len(content)

# print content