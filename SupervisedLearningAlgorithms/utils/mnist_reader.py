import numpy as np
import pickle


# Path to file
data_path = 'data/'

def load_mnist_dataset(file):
    '''First time the file is requested it will read the file and save it as a pickle file,
        times after that the file will be loaded from the directory \"data/filename.pkl\" as an numpy array'''
    try:
        filename = file.split('.')
        return collect_file(filename[0])
    except FileNotFoundError: 
        print('Writing file...')
        data = np.loadtxt('data/'+file, delimiter=',')
        with open(data_path+'MNIST_'+filename[0]+'.pkl', 'wb') as fp : 
            pickle.dump(data, fp)
        return collect_file(filename[0])
    
   
def collect_file(filename):
    with open(data_path+'MNIST_'+filename+'.pkl', 'rb') as fp : 
        print('Loading data...')
        as_nparray = pickle.load(fp)
    return as_nparray


        
