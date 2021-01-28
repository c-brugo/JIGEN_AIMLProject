import numpy as np
from PIL import Image
#from pathlib import Path
from random import randint
#import argparse
from tqdm import trange
import itertools
from scipy.spatial.distance import cdist


class DatasetEdit:

    #slice_size
    #directory
    #filename # = outname
    #best_hamming
    #M
    #N
    #pos_auto_x
    #pos_auto_y
    #used_pos
    
    
    def __init__(self,slice_size,img_dim):
        self.slice_size=slice_size
        #self.directory=directory

        #img = Image.open(directory+'pic_001.jpg') # access a picture to retrieve size
        
        
        self.M = img_dim//slice_size
        self.N = img_dim//slice_size

        self.mappa3=[[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2]]
        self.mappa2=[[0,0],[1,0],[0,1],[1,1]]
        self.mappa4=[[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2],[0,3],[1,3],[2,3],[3,3]]

        self.pos_auto_x = np.arange(0,img_dim,self.M)
        self.pos_auto_y = np.arange(0,img_dim,self.N)#[y for y in range(0,img_dim,N)]

        self.diff_pos = slice_size^2 # slice_size = 3 so different positions are 9 (3^2) / 2 so 4 / 4 so 16

        self.permutations()
    
    


    #used to retrive where to put a sliced image in mosaic.
    def matrixIndeces(self,index,case):
        if case == 1: #2x2
            ind_y = self.mappa2[index][1] # row
            ind_x = self.mappa2[index][0] # column  
        elif case == 2: #3x3
            ind_y = self.mappa3[index][1] # row
            ind_x = self.mappa3[index][0] # column
        elif case == 3: #4x4
            ind_y = self.mappa4[index][1] # row
            ind_x = self.mappa4[index][0] # column
        else:
            # default case
            pass
        
        return ind_x,ind_y

    def permutations(self):
        outname = 'permutations/permutations_hamming_%d'%(self.slice_size)
        selection = "max"
        # with [0,1,2,3,4,5,6,7,8] - original image
        P_hat_full = np.array(list(itertools.permutations(list(range(self.slice_size), self.slice_size))))
        P_hat = np.delete(P_hat_full,0,0)
        n = P_hat.shape[0]
        
        for i in trange(30):
            if i==0:
                j = np.random.randint(n)
                P = np.array(P_hat[j]).reshape([1,-1])
            else:
                P = np.concatenate([P,P_hat[j].reshape([1,-1])],axis=0)
            
            P_hat = np.delete(P_hat,j,axis=0)
            D = cdist(P,P_hat, metric='hamming').mean(axis=0).flatten()
            
            j = D.argmax()
            
            if selection=='max':
                j = D.argmax()
            else:
                m = int(D.shape[0]/2)
                S = D.argsort()
                j = S[np.random.randint(m-10,m+10)]
            
            #if i%30==0:
                #np.save(outname,P)
        #np.save(outname,P)
        self.filename = outname
        self.best_hamming = P
        #print('file created --> '+outname)

    def readHamming(self):
        
        best_hamming = np.load(self.filename+'.npy')
        
        return best_hamming #access as matrix

    def edit(self, img):
        """ slice the image in parts
            Args:
                img (Image): opened image file
                img_name (str): used to save the edited image
                case (int): slice type 1 = 2x2, 2 = 3x3, 3 = 4x4
        """
        # Open images and store them in a list
        #for path in Path(self.directory).rglob('*.jpg'): # errpr: WindowsPath object is not iterable
        #images = [Image.open(x) for x in path]
        
        
        #slice in parts and create the new image
        #for img in images:
        k=0 # index for hamming_array
        i = randint(0, 29) # 1 random best hamming
        hamming_array=self.best_hamming[i]
        # create a new image with the appropriate height and width
        new_img = Image.new('RGB', (img.size[1], img.size[0]))
        for y in range(0,img.size[0],self.M):
            for x in range(0,img.size[1],self.N):
                tile = img.crop((x,y,x+self.M,y+self.N))


             #place tile
            if k>=self.diff_pos:
                break
            MatInd = self.matrixIndeces(hamming_array[k],self.slice_size)
            i=int(MatInd[0])
            j=int(MatInd[1])
            new_img.paste(tile, (self.pos_auto_x[i],self.pos_auto_y[j]))
            k=k+1

        # Save the image
        #img_name=path.name.split('.')
        #full_img_name=img_name+"_sliced(%d)"+".jpg"%(i+1) # i is the label (1 to 30)
       #new_img.save(full_img_name)
        return new_img,i+1

    def resize(self, im,size_width,size_height) -> Image:
    
        new_dim = (size_width, size_height)
        return im.resize(new_dim)
