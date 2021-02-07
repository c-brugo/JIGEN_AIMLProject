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
    #P: n of permutations
    #pos_auto_x
    #pos_auto_y
    #used_pos
    
    
    def __init__(self, img_dim, P, slice_size=3):
        self.slice_size=slice_size
        #self.directory=directory

        #img = Image.open(directory+'pic_001.jpg') # access a picture to retrieve size
        self.img_input_nn_dim = img_dim

        if img_dim%slice_size != 0:
            self.img_dim = 224
        else:
            self.img_dim = img_dim
        
        self.M = self.img_dim//slice_size
        self.N = self.img_dim//slice_size

        self.mappa3=[[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2]]
        self.mappa2=[[0,0],[1,0],[0,1],[1,1]]
        self.mappa4=[[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2],[0,3],[1,3],[2,3],[3,3]]

        self.pos_auto_x = np.arange(0,self.img_dim,self.M)
        self.pos_auto_y = np.arange(0,self.img_dim,self.N)#[y for y in range(0,img_dim,N)]

        self.diff_pos = slice_size**2 # slice_size = 3 so different positions are 9 (3^2) / 2 so 4 / 4 so 16
        self.P = P

        self.permutations()
    
    


    #used to retrive where to put a sliced image in mosaic.
    def matrixIndeces(self,index,slice_size):
        if slice_size == 2: #2x2
            ind_y = self.mappa2[index][1] # row
            ind_x = self.mappa2[index][0] # column  
        elif slice_size == 3: #3x3
            ind_y = self.mappa3[index][1] # row
            ind_x = self.mappa3[index][0] # column
        elif slice_size == 4: #4x4
            ind_y = self.mappa4[index][1] # row
            ind_x = self.mappa4[index][0] # column
        else:
            # default case
            pass
        
        return ind_x,ind_y

    def permutations(self):
        outname = 'permutations/permutations_hamming_%d'%(self.diff_pos)
        selection = "max"
        # with [0,1,2,3,4,5,6,7,8] - original image
        P_hat_full = np.array(list(itertools.permutations(list(range(self.diff_pos)), self.diff_pos)))
        P_hat = np.delete(P_hat_full,0,0)
        
        n = P_hat.shape[0]

        for i in range(self.P):
            if i==0:
                j = np.random.randint(n)
                P = np.array(P_hat[j]).reshape([1,-1])
            else:
                P = np.concatenate([P,P_hat[j].reshape([1,-1])],axis=0)
            
            P_hat = np.delete(P_hat,j,axis=0)
            D = cdist(P,P_hat, metric='hamming').mean(axis=0).flatten()
            
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
        """
        if self.img_dim!=self.img_input_nn_dim:
            img = self.resize(img, self.img_dim, self.img_dim)
        
        # Open images and store them in a list
        #for path in Path(self.directory).rglob('*.jpg'): # errpr: WindowsPath object is not iterable
        #images = [Image.open(x) for x in path]
        
        #if img.size[0]%self.slice_size!=0 or img.size[0]%self.slice_size!=0:

        #slice in parts and create the new image
        #for img in images:
        k=0 # index for hamming_array
        perm = randint(0, self.P-1) # 1 random best hamming
        hamming_array=self.best_hamming[perm]
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
        if self.img_dim!=self.img_input_nn_dim:
            new_img = self.resize(new_img, self.img_input_nn_dim, self.img_input_nn_dim)
        return new_img, perm+1
    
    def odd_one_out(self, img1, img2):
        """ switch a random tile in puzzle of the first image 
            with a random crop of the second image
            Args:
                img1 (Image): opened image file
                img2 (Image): opened image file
            Out:
                img1 (Image): image with odd tile
                index_odd (int): position of the odd tile
        """

        img1, _ = self.edit(img1)
        
        # random crop in img2
        x = randint(0, img2.size[0] - self.M)
        y = randint(0, img2.size[1] - self.N)
        odd_tile = img2.crop((x,y,x+self.M,y+self.N))
        
        # random position of the odd tile in img1
        index_odd = randint(0,8)
        
        # paste the odd tile in img1
        MatInd = self.matrixIndeces(index_odd,self.slice_size)
        i = int(MatInd[0])
        j = int(MatInd[1])
        img1.paste(odd_tile, (self.pos_auto_x[i],self.pos_auto_y[j]))

        return img1, index_odd+1
    
    def randomRotation(self, img):
        """ rotate the image of a random multiple of 90 degree 
            Args: 
                img (Image): opened image file
            Out:  
                img (Image): rotated image
                angle (int): angle of rotation
        """
        
        angle = randint(0, 2)
        
        if angle == 0:
            new_img = img.transpose(Image.ROTATE_90) 
        elif angle == 1:
            new_img = img.transpose(Image.ROTATE_180)
        elif angle == 2:
            new_img = img.transpose(Image.ROTATE_270)
                
        return new_img, angle+1

    def resize(self, im, size_width, size_height) -> Image:
        new_dim = (size_width, size_height)
        return im.resize(new_dim)
