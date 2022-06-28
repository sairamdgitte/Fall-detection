import numpy as np
import os


list_of_files = []
for i in os.listdir('./Float'):
    
    for j in os.listdir('./Float/{}'.format(i)):
        list_of_files = []
        for k in os.listdir('./Float/{}/{}'.format(i,j)):
            list_of_files.append(int(k.split('.')[0]))
        # remaining_frames = 400 - max(list_of_files)
        # print(max(list_of_files))
        for rem in range(max(list_of_files)+1, 400):

            keypoints = np.zeros(258)
            npy_path = os.path.join('Float', i, j, str(rem))
            np.save(npy_path, keypoints)

    print('Done...')    
        


     
    