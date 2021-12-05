from lithostl import lithophane_rev

#### Create Lithophane from Chicken
lith = lithophane_rev(picture = 'example/Chicken.jpg', background = '', sigma = 0, picture_height = 100, picture_width = 0,
                        sphere_radius = 50, cylinder_radius = 25, cylinder_height = 10,
                        lithodepth = 3, lithominheight = 1)

#### Create Coordinates for STL Model
lith.generateCoordinates()

#### save STL-File
lith.save_stl('example/', append_str = '')