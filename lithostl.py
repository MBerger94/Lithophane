#!/usr/bin/env python

import numpy as np
import matplotlib.image as img
from skimage.transform import resize
from tqdm import tqdm
from stl import mesh
import argh

class lithophane():

    def __init__(self, filename, typestr, width = None):
        self.pic_filename = filename
        self._lith_type   = typestr
        if width == None:
            self.image_width  = None
        else:
            self.image_width  = width
        
        self.model      = None
        self.image      = None
        self.grayImage  = None
        self._xyz       = None
        self._front     = None
        self._back      = None
        self._depth     = None
        self._offset    = None

        self.readImage()

    def readImage(self):
        self.image = img.imread(self.pic_filename)

    def rgb_to_gray(self):
        r, g, b = self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def scale_image(self):
        """
        Scale image width
        10 pixels = 1 mm
        """

        ydim = self.image.shape[0]
        xdim = self.image.shape[1]

        scale = (self.image_width * 10 / xdim)
        newshape = (int(ydim * scale), int(xdim * scale), 3)
        self.image = resize(self.image, newshape)

    def generateSTL(self, depth = 3.0, offset = 0.5, filter_sigma = 0):
        self._depth = depth
        self._offset = offset

        if self.image_width == None:
            self.image_width = self.image.shape[1]

        print("Building STL with width " + str(self.image_width) + "mm.")
        
        self.scale_image()
        self.image = self.image / np.max(self.image)
        self.grayImage = self.rgb_to_gray()
        # Invert z matrix
        ngray = 1 - np.double(self.grayImage)

        # scale z matrix
        z_middle = ngray * depth + offset

        # apply Gaussian filter to z matrix
        if filter_sigma > 0:    
            from scipy.ndimage import gaussian_filter as gsflt
            z_middle = gsflt(z_middle, sigma = filter_sigma)
        
        # add border of zeros to help with back.
        z = np.zeros([z_middle.shape[0] + 2, z_middle.shape[1] + 2])
        z[1: -1, 1: -1] = z_middle


        x1 = np.linspace(1, z.shape[1] / 10, z.shape[1])
        y1 = np.linspace(1, z.shape[0] / 10, z.shape[0])

        x, y = np.meshgrid(x1, y1)

        x = np.fliplr(x)

        self._xyz = [x, y, z]

    def generateModel(self):
        x,   y,  z = self._front
        bx, by, bz = self._back

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


    def _makeSphere(self, BottomHole = 0.85, TopFading = 10, ScalingFunc = 'exp', FlatBottomBrim = False):
        """
        BottomHole:     FLOAT,    % of Bottom Opening
        TopFading:      INT,      % to add to Sphere for Picture Fading
        ScalingFunc:    STR,      Scaling Function for Fading
        """
        x,y,z = self._xyz

        def fading_function(num, ScalingFunc):
            out = np.zeros(num)
            ### Linear
            if ScalingFunc == 'None':
                out = np.ones(num)
            if ScalingFunc == 'linear':
                out = np.arange(num) / (num)
            ### Exponential
            if ScalingFunc == 'exp':
                out = np.exp(-7 * (1 - np.arange(num) / (num)))
            if ScalingFunc == 'poly':
                out = -(1 - np.arange(num) / num)**1.7 + 1
            return out

        Nx_Top = int(x.shape[0] * TopFading / 100)
        radius = (np.max(x) - np.min(x)) / (2 * np.pi) * (1 + TopFading / 100)
        diameter = 2 * radius

        x_range = x.shape[0] + Nx_Top

        front_x = np.zeros((x_range, x.shape[1]))
        front_y = np.zeros((x_range, y.shape[1]))
        front_z = np.zeros((x_range, z.shape[1]))
        back_x  = np.zeros((x_range, x.shape[1]))
        back_y  = np.zeros((x_range, y.shape[1]))
        back_z  = np.zeros((x_range, z.shape[1]))

        front_x [Nx_Top:, :] = x.copy()
        front_y [Nx_Top:, :] = y.copy()
        front_z [Nx_Top:, :] = z.copy()
        back_x  [Nx_Top:, :] = x.copy()
        back_y  [Nx_Top:, :] = y.copy()
        back_z  [Nx_Top:, :] = z.copy()

        for c in range(0, x.shape[1]):
            if TopFading > 0:
                front_z[:Nx_Top + 2, c] = z[2, c] * fading_function(Nx_Top + 2, ScalingFunc)

        print("Expected Sphere Diameter %.2f mm" % diameter)

        ph = np.array([np.min([float(c) / x_range, BottomHole]) for c in range(0, x_range)]) * np.pi
        th = np.array([c / (x.shape[1] - 10) * 2 * np.pi for c in range(0, x.shape[1])])
        phs, ths = np.meshgrid(ph, th, indexing = 'ij')
        rs  = radius + front_z
        
        ## short bottom for flat bottom
        if FlatBottomBrim:
            rs[phs > np.pi * (BottomHole - 0.01)] = radius + self._depth + self._offset
            print(180 - (BottomHole - 0.01) * 180)
            print("Bottom Hole Radius %.2f mm" % (radius * np.sin(np.pi * (1 - BottomHole + 0.01))))

        front_x = rs * np.cos(ths) * np.sin(phs)
        front_y = rs * np.cos(phs)
        front_z = rs * np.sin(ths) * np.sin(phs)
        back_x  = radius * np.cos(ths) * np.sin(phs)
        back_y  = radius * np.cos(phs)
        back_z  = radius * np.sin(ths) * np.sin(phs)

        self._front = (front_x, front_y, front_z)
        self._back  = (back_x, back_y, back_z)

    def _makeCylinder(self, **kwargs):
        x, y, z = self._xyz

        front_x = x.copy()
        front_y = y.copy()
        front_z = z.copy()
        back_x  = x.copy()
        back_y  = y.copy()
        back_z  = z.copy()

        radius = (np.max(x) - np.min(x)) / (2 * np.pi)

        th = np.array([c / (x.shape[1] - 10) * 2 * np.pi for c in range(0, x.shape[1])])
        rs = radius + front_z

        ths, _ = np.meshgrid(th, np.ones(x.shape[0]), indexing = 'xy')

        print(f"Expected Cylinder Diameter {2 * radius}mm")
        
        front_x = rs * np.cos(ths)
        front_z = rs * np.sin(ths)
        back_x = radius * np.cos(ths)
        back_z = radius * np.sin(ths)

        self._front = (front_x, front_y, front_z)
        self._back  = (back_x, back_y, back_z)

    def _makePlane(self, **kwargs):
        back_x = self._xyz[0].copy()
        back_y = self._xyz[1].copy()
        back_z = np.zeros(self._xyz[2].shape)

        self._front = self._xyz
        self._back  = (back_x, back_y, back_z)

    def generateCoordinates(self, **kwargs):
        if self._xyz == None:
            print("STL of image needed to generate Coordinates.")
            print("Run generateSTL()!")
            exit()

        if self._lith_type == 'sphere':
            self._makeSphere(**kwargs)
            print("Model Coordinates generated!")
        elif self._lith_type == 'cylinder':
            self._makeCylinder(**kwargs)
            print("Model Coordinates generated!")
        elif self._lith_type == 'flat':
            self._makePlane(**kwargs)
            print("Model Coordinates generated!")
        else:
            print("Type not supported!")
            pass

    def save_stl(self, location, append_str = ''):
        if append_str == '':
            out = location + self.pic_filename.split('/')[-1].split('.')[0] + '.stl'
        else:
            out = location + self.pic_filename.split('/')[-1].split('.')[0] + '_' + append_str + '.stl'
        self.model.save(out)
        print("Model saved to " + out + "!")

    @property
    def lith_type(self):
        print(self._lith_type)

    @lith_type.setter
    def lith_type(self, typestr):
        self._lith_type = typestr

    def plotModel(self, show = True):
        if self.model == None:
            print("Model is empty.")
        else:
            from mpl_toolkits import mplot3d
            import matplotlib.pyplot as plt

            # Create a new plot
            figure = plt.figure()
            axes = mplot3d.Axes3D(figure)

            # Load the STL files and add the vectors to the plot
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.model.vectors))

            # Auto scale to the mesh size
            scale = self.model.points.flatten()
            axes.auto_scale_xyz(scale, scale, scale)

            if show == True:
                plt.show()
            else:
                plt.savefig('Model.jpg')

    def plotImage(self, show = True):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2)

        ax[0].imshow(self.image)
        ax[1].imshow(self._xyz[2])

        if show == True:
            plt.show()
        else:
            plt.savefig('Images.jpg')


def main(picture = '', form = '', width = 10.0, hrange = 2.5, minheight = 0.5, gfilter = 0.0, scaling = 'None', topfading = 0.0, output = './', appendix = ''):
    """
    Lithophane

    picture     <STR> ...   Directory to store output
    form        <STR> ...   Sphere, Cylinder or Flat
    width       <FLT> ...   Scale Picture to Width (10 pixels = 1 mm)
    hrange      <FLT> ...   Height Profile minheight + grayscale * hrange
    minheight   <FLT> ...   Height Profile minheight + grayscale * hrange
    gfilter     <FLT> ...   Gaussian Filter Width
    scaling     <STR> ...   Scaling Function for Top Fading (Sphere only)
    topfading   <FLT> ...   % to add to top for fading (Sphere only)
    output      <STR> ...   Output Directory
    apendix     <STR> ...   Append String to Output
    """

    #### Create Lithophane from Chicken
    lith = lithophane(picture, form, width = width)

    #### Create Image(xyz) and apply gaussian filter with sigma = 3
    lith.generateSTL(depth = hrange, offset = minheight, filter_sigma = gfilter)

    #### Create Coordinates for STL Model
    lith.generateCoordinates(ScalingFunc = scaling, TopFading = topfading)

    #### Generate STL from coordinates
    lith.generateModel()

    #### save STL-File
    lith.save_stl(output, append_str = appendix)

if __name__ == '__main__':
    argh.dispatch_command(main)