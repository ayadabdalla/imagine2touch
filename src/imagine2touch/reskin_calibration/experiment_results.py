import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2

# import dataset


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("pass path to experiment directory, experiment name and experiment_n")
        exit()

    # force = np.load(f'{sys.argv[1]}/force', mmap_mode='r')
    # # print(force)

    # pose = np.load(f'{sys.argv[1]}/pose', mmap_mode='r')
    # print(pose.shape)

    reskin_reading = np.load(
        f"{sys.argv[1]}/{sys.argv[2]}_tactile/{sys.argv[3]}_reskin", allow_pickle=True
    )
    reskin_reading = np.squeeze(reskin_reading)[
        :, 2
    ]  # extract lists of magnetometers values and temperatures as array of lists
    reskin_reading = list(reskin_reading)  # convert to list of lists then to nd array
    reskin_reading = np.asarray(reskin_reading)
    reskin_reading = np.delete(
        reskin_reading, [0, 4, 8, 12, 16], 1
    )  # eliminate temperatures
    inputs = reskin_reading
    print(inputs.shape)

    # center=np.expand_dims(np.expand_dims(inputs[:,0:3],1),2)
    # top=np.expand_dims(np.expand_dims(inputs[:,3:6],1),2)
    # right=np.expand_dims(np.expand_dims(inputs[:,6:9],1),2)
    # bottom=np.expand_dims(np.expand_dims(inputs[:,9:12],1),2)
    # left=np.expand_dims(np.expand_dims(inputs[:,12:15],1),2)
    # zeros=np.zeros((1,1,3))

    # top_row=np.hstack((zeros,top[9]))
    # top_row=np.hstack((top_row,zeros))
    # middle_row=np.hstack((right[9],center[9]))
    # middle_row=np.hstack((middle_row,left[9]))
    # bottom_row=np.hstack((zeros,bottom[9]))
    # bottom_row=np.hstack((bottom_row,zeros))
    # current_image=np.vstack((top_row,middle_row,bottom_row))

    # print(current_image.shape)
    # cv2.imshow("reskin_image",current_image)
    # cv2.waitKey(0)

    # ambient_reading = np.load(f'{sys.argv[1]}/{sys.argv[2]}_tactile/{sys.argv[3]}_reskin_ambient', allow_pickle=True)
    # processed_ambient_readings =[]
    # for readings in ambient_reading: # average sets of ambient readings
    #     readings = np.squeeze(readings)[:,2] # extract lists of magnetometers values and temperatures as array of lists
    #     readings = list(readings) # convert to list of lists then to nd array
    #     readings = np.asarray(readings)
    #     readings = np.delete(readings,[0,4,8,12,16],1) # eliminate temperatures
    #     processed_ambient_readings.append(np.mean(readings,axis=0))
    # inputs = np.asarray(processed_ambient_readings)
    # inputs=np.asarray(ambient_reading)
    # print(inputs.shape)

    # inputs, targets = dataset.prepare_reskin_data(sys.argv[1],sys.argv[2])
    # inputs_binary, targets_binary = dataset.prepare_reskin_data(sys.argv[3],sys.argv[4])
    # # inputs = np.vstack((inputs,inputs_binary))
    # inputs= reskin_reading
    # print(ambient_reading.shape)
    # print(inputs.mean(axis=0))

# Trim last rubbish readings
# print(len(ambient_reading[16]))
# with open(f'{sys.argv[1]}/ambient_readings','wb') as readings :
#     np.save(readings,ambient_reading[:2])
# with open(f'{sys.argv[1]}/ambient_readings','wb') as readings :
#     np.save(readings,reskin_reading[:4835])
# with open(f'{sys.argv[1]}/force','wb') as force :
#     np.save(force,force[:4835])
# print('done with saving force')
# with open(f'{sys.argv[1]}/pose','wb') as pose :
#     np.save(pose,pose[:4835])

# Plotting
# fig1 = plt.figure()
# # #magnetometer 1
# plt.plot(inputs[:,0])
# plt.plot(inputs[:,1])
# plt.plot(inputs[:,2])
# plt.legend(['bx1','by1','bz1'])

# # # #magnetometer 2
# fig2 = plt.figure()


# plt.plot(inputs[:,3])
# plt.plot(inputs[:,4])
# plt.plot(inputs[:,5])
# plt.legend(['bx2','by2','bz2'])


# # # # #magnetometer 3
# fig3 = plt.figure()

# plt.plot(inputs[:,6])
# plt.plot(inputs[:,7])
# plt.plot(inputs[:,8])
# plt.legend(['bx3','by3','bz3'])


# # # # #magnetometer 4
# fig4 = plt.figure()

# plt.plot(inputs[:,9])
# plt.plot(inputs[:,10])
# plt.plot(inputs[:,11])
# plt.legend(['bx4','by4','bz4'])


# # # # #magnetometer 5
# fig5 = plt.figure()

# plt.plot(inputs[:,12])
# plt.plot(inputs[:,13])
# plt.plot(inputs[:,14])

# plt.legend(['bx5','by5','bz5'])

# plt.show()
