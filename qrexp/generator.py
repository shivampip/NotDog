import pyqrcode 
from pyqrcode import QRCode 
import numpy as np


def makeqr(data):
    qr= pyqrcode.create(data, mode= 'numeric')
    qr.png("data/{}.png".format(str(data)), scale= 4)
    print("{} created".format(data))



def generate_random(total= 100):
    for i in range(total):
        num= np.random.randint(1000, 9999)
        makeqr(num)


def generate_given(arr):
    for a in arr:
        makeqr(a) 
