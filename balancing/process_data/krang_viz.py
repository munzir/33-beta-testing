import matplotlib.pyplot as plt
import numpy as np



# Read Data

with open("krang_run25") as f:
    lines = f.readlines()

raw_data = [x.strip() for x in lines]

rows = len(raw_data)
cols = len(raw_data[0].split())

data = np.zeros((rows,cols))

for i,row_space in enumerate(raw_data):
    row = row_space.split()
    data[i,:] =  row

time = np.cumsum(data[:,0])


plt.figure()

titles = ["dt","leftWheel","rightWheel","state0","state1","state2","state3","state4","state5"]

for i,title in enumerate(titles):
    plt.figure()
    # plt.subplot(2,1,i)
    plt.plot(time,data[:,i])
    plt.title(title)

plt.show()

print data
print rows, cols





# print line

# print raw_data

#     for xx in x:
#         content.append(xx)

# content = [x.strip() for x in content]

# print len(content)

# print content