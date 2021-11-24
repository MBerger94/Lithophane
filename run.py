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


#### Change Build - Form of Lithophane
lith.lith_type = "sphere"

#### Build STL from Chicken now as Sphere
lith.generateSTL()
lith.generateCoordinates()
lith.generateModel()
lith.save_stl("./example/", append_str = 'sphere')