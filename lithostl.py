#!/usr/bin/env python

import numpy as np
import matplotlib.image as img
from skimage.transform import resize
from tqdm import tqdm
from stl import mesh
import argh

class lithophane_rev():

    def __init__(self, picture, background, sphere_radius = 50, cylinder_radius = 25, cylinder_height = 10, 
                lithodepth = 3, lithominheight = 1, 
                sigma = 0, picture_height = 100, picture_width = 0):
        #### units are all mm
        self.picture_dir = picture
        self.background_dir = background

        self.SphR   = sphere_radius

        self.CyR    = cylinder_radius
        self.CyH    = cylinder_height

        self.LiD    = lithodepth
        self.LiH    = lithominheight

        self.FiSig  = sigma

        self.picture = self.readImage(self.picture_dir)
        if background != '':
            self.backpic = self.readImage(self.background_dir)

        self.PiH    = picture_height
        self.PiW    = picture_width

        if self.PiW == 0:
            self.PiW = 2 * np.pi * self.SphR

        self.picture = self.scale_image(self.picture, self.PiH)
        if background != '':
            self.backpic = self.scale_image(self.backpic, np.pi * self.SphR)
        else:
            self.backpic = None

        self._check_sanity()

        self.model = None

    def _check_sanity(self):
        if self.SphR < self.CyR:
            print("Sphere must have larger Radius than Cylinder base!")
            exit()

        if self.PiH > np.pi * self.SphR:
            print("Image Height cannot be larger than Half Sphere Circumference!")
            exit()

        if self.PiW > 2 * np.pi * self.SphR:
            print("Image Width cannot be larger than Sphere Cirumference!")

    def readImage(self, picdir):
        return img.imread(picdir)

    def rgb_to_gray(self, pic):
        r, g, b = pic[:, :, 0], pic[:, :, 1], pic[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def scale_image(self, pic, height):
        """
        Scale image width
        10 pixels = 1 mm
        """

        ydim = pic.shape[0]
        xdim = pic.shape[1]

        scale = (height * 10 / xdim)
        newshape = (int(ydim * scale), int(xdim * scale), 3)
        return resize(pic, newshape)
        
    def RevFunc(self, x):
        intersec = self.SphR + np.sqrt(self.SphR**2 - self.CyR**2)

        if 0 <= x < intersec:
            ## Sphere
            return np.sqrt(self.SphR**2 - (x - self.SphR)**2)
        elif intersec <= x <= intersec + self.CyH:
            ## Cylinder
            return self.CyR

    def generateSTL(self, pic, sigma, lith_d):

        pic = pic / np.max(pic)
        grayImage = self.rgb_to_gray(pic)
        # Invert z matrix
        ngray = 1 - np.double(grayImage)

        # scale z matrix
        z = ngray * lith_d

        # apply Gaussian filter to z matrix
        if sigma > 0:    
            from scipy.ndimage import gaussian_filter as gsflt
            z = gsflt(z, sigma = sigma)
        
        # add border of zeros to help with back.
        # z = np.zeros([z_middle.shape[0] + 2, z_middle.shape[1] + 2])
        # z[1: -1, 1: -1] = z_middle


        x1 = np.linspace(1, z.shape[1] / 10, z.shape[1])
        y1 = np.linspace(1, z.shape[0] / 10, z.shape[0])

        x, y = np.meshgrid(x1, y1)

        x = np.fliplr(x)

        return x, y, z

    def generateCoordinates(self):

        vRevFunc = np.vectorize(self.RevFunc)

        x, y, z = self.generateSTL(self.picture, self.FiSig, self.LiD)

        if type(self.backpic) == np.ndarray:
            bx, by, bz = self.generateSTL(self.backpic, 0, self.LiD / 3)
        else:
            bz = 0

        intersec = self.SphR + np.sqrt(self.SphR**2 - self.CyR**2)
        phs = np.linspace(0, 2 * np.pi, num = z.shape[1] + int((2 * np.pi * self.SphR - self.PiW) * 10))
        xs  = np.linspace(0, intersec + self.CyH, num = z.shape[0] + int((np.pi * self.SphR - self.PiH) * 10) + int(self.CyH * 10))

        phsgrid, xsgrid = np.meshgrid(phs, xs)
        rsgrid = vRevFunc(xsgrid)

        back_x = np.cos(phsgrid) * (rsgrid - self.LiH)
        back_y = np.sin(phsgrid) * (rsgrid - self.LiH)
        back_z = xsgrid

        xStart = int((np.pi * self.SphR - self.PiH) * 5)
        yStart = int((2 * np.pi * self.SphR - self.PiW) * 5)
        
        rsgrid[xStart:xStart + z.shape[0], yStart:yStart + z.shape[1]] = rsgrid[xStart:xStart + z.shape[0], yStart:yStart + z.shape[1]] + z

        # TODO: Add Background Image

        front_x = np.cos(phsgrid) * (rsgrid)
        front_y = np.sin(phsgrid) * (rsgrid)
        front_z = xsgrid

        self.generateModel([front_x, front_y, front_z], [back_x, back_y, back_z])

    def generateModel(self, front, back):
        x,   y,  z = front
        bx, by, bz = back

        count = 0
        points = np.zeros((((z.shape[0] - 1) * (z.shape[1] - 1) + (bz.shape[0] - 1) * (bz.shape[1] - 1)) + 1, 6, 3))
        triangles = []
        bar = tqdm(total = (z.shape[0] - 1) * (z.shape[1] - 1), unit = 'Front Points')
        for i in range(z.shape[0] - 1):
            for j in range(z.shape[1] - 1):

                # Triangle 1
                points[i * (z.shape[1] - 1) + j, 0] = [x[i][j]    , y[i][j]    , z[i][j]]
                points[i * (z.shape[1] - 1) + j, 1] = [x[i][j + 1], y[i][j + 1], z[i][j + 1]]
                points[i * (z.shape[1] - 1) + j, 2] = [x[i + 1][j], y[i + 1][j], z[i + 1][j]]

                triangles.append([count, count + 1, count + 2])

                # Triangle 2
                points[i * (z.shape[1] - 1) + j, 3] = [x[i][j + 1]    , y[i][j + 1]    , z[i][j + 1]]
                points[i * (z.shape[1] - 1) + j, 4] = [x[i + 1][j + 1], y[i + 1][j + 1], z[i + 1][j + 1]]
                points[i * (z.shape[1] - 1) + j, 5] = [x[i + 1][j]    , y[i + 1][j]    , z[i + 1][j]]

                triangles.append([count + 3, count + 4, count + 5])

                count += 6
                bar.update(1)
        bar.close()
        idx0 = (z.shape[0] - 1) * (z.shape[1] - 1) + 1
        # BACK
        bar = tqdm(total = (bz.shape[0] - 1) * (bz.shape[1] - 1), unit = 'Back Points')
        for i in range(bz.shape[0] - 1):
            for j in range(bz.shape[1] - 1):

                # Triangle 1
                points[idx0 + i * (bz.shape[1] - 1) + j, 0] = [bx[i + 1][j], by[i + 1][j], bz[i + 1][j]]
                points[idx0 + i * (bz.shape[1] - 1) + j, 1] = [bx[i][j + 1], by[i][j + 1], bz[i][j + 1]]
                points[idx0 + i * (bz.shape[1] - 1) + j, 2] = [bx[i][j]    , by[i][j]    , bz[i][j]]

                triangles.append([count, count + 1, count + 2])

                # Triangle 2
                points[idx0 + i * (bz.shape[1] - 1) + j, 3] = [bx[i + 1][j]    , by[i + 1][j]    , bz[i + 1][j]]
                points[idx0 + i * (bz.shape[1] - 1) + j, 4] = [bx[i + 1][j + 1], by[i + 1][j + 1], bz[i + 1][j + 1]]
                points[idx0 + i * (bz.shape[1] - 1) + j, 5] = [bx[i][j + 1]    , by[i][j + 1]    , bz[i][j + 1]]

                triangles.append([count + 3, count + 4, count + 5])

                count += 6
                bar.update(1)
        bar.close()

        # TODO bottom
        pts = points.reshape(((z.shape[0] - 1) * (z.shape[1] - 1) + (bz.shape[0] - 1) * (bz.shape[1] - 1) + 1) * 6, 3)
        # Create the mesh
        model = mesh.Mesh(np.zeros(len(triangles), dtype = mesh.Mesh.dtype))
        for i, f in enumerate(triangles):
            for j in range(3):
                model.vectors[i][j] = pts[f[j]]

        self.model = model

    def save_stl(self, location, append_str = ''):
        if append_str == '':
            out = location + self.picture_dir.split('/')[-1].split('.')[0] + '.stl'
        else:
            out = location + self.picture_dir.split('/')[-1].split('.')[0] + '_' + append_str + '.stl'
        self.model.save(out)
        print("Model saved to " + out + "!")
        
# l = revolution_lithophane('../Unbenannt4.jpg')

# l.generateCoordinates()
# l.save_stl('./examples', append_str = 'Rev-Sig-3')


def main(picture = '', background = '', radius = 50, base_rad = 25, base_height = 10, litho_depth = 3, litho_min_height = 1, sigma = 0, picture_height = 100, picture_width = 0, output = './', appendix = ''):
    
    #### Create Lithophane from Chicken
    lith = lithophane_rev(picture, background = background, sigma = sigma, picture_height = picture_height, picture_width = picture_width,
                            sphere_radius = radius, cylinder_radius = base_rad, cylinder_height = base_height,
                            lithodepth = litho_depth, lithominheight = litho_min_height)

    #### Create Coordinates for STL Model
    lith.generateCoordinates()

    #### save STL-File
    lith.save_stl(output, append_str = appendix)

if __name__ == '__main__':
    argh.dispatch_command(main)