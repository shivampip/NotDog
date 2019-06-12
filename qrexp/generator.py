import pyqrcode 
from pyqrcode import QRCode 
import numpy as np
import random
import os

def makeqr(data, path='qrexp/data/'):
    qr= pyqrcode.create(data, mode= 'numeric')
    qr.png("{}{}.png".format(path, str(data)), scale= 4)
    print("{} created".format(data))



def generate_random(path='qrexp/data/', total= 100):
    if(not os.path.exists(path)):
        os.makedirs(path)
    nums= random.sample(range(1000,9999), total)
    for num in nums:
        #num= np.random.randint(1000, 9999)
        makeqr(num, path= path)


def generate_given(arr):
    for a in arr:
        makeqr(a) 
