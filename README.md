# Litho-STL
Python Class to create Lithophanes from Images and export them as STL-File.

## Dependencies
- python >= 3.6.9
- numpy >= 1.19.5
- numpy-stl
- tqdm >= 4.42.1
- skimage >= 0.17.2
- matplotlib >= 3.1.3
> optional
- scipy >= 1.3.1

## lithophane Class
`generateSTL(self, depth = 3.0, offset = 0.5, filter_sigma = 0)`
- depth
- offset
- filter_sigma: sigma for Gaussian Filtering

`generateCoordinates()`

`generateModel()`

# Example
```python
from lithostl import lithophane

#### Create Lithophane from Chicken
lith = lithophane("./example/Chicken.jpg", "flat", width = 40)

#### Create Image(xyz) and apply gaussian filter with sigma = 3
lith.generateSTL(filter_sigma = 3)

#### plot Original Image and gray scale Image
lith.plotImage(show = False)

#### Create Coordinates for STL Model
lith.generateCoordinates()

#### Generate STL from coordinates
lith.generateModel()

#### save STL-File
lith.save_stl("./example/", append_str = 'flat')
```