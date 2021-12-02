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
Running `python lithostl.py -h` creates the following output:
```bash
usage: lithostl.py [-h] [-p PICTURE] [-f FORM] [-w WIDTH] [--hrange HRANGE] [-m MINHEIGHT] [-g GFILTER] [-s SCALING] [-t TOPFADING] [-o OUTPUT] [-a APPENDIX]

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


optional arguments:
  -h, --help            show this help message and exit
  -p PICTURE, --picture PICTURE
                        ''
  -f FORM, --form FORM  ''
  -w WIDTH, --width WIDTH
                        10.0
  --hrange HRANGE       2.5
  -m MINHEIGHT, --minheight MINHEIGHT
                        0.5
  -g GFILTER, --gfilter GFILTER
                        0.0
  -s SCALING, --scaling SCALING
                        'None'
  -t TOPFADING, --topfading TOPFADING
                        0.0
  -o OUTPUT, --output OUTPUT
                        './'
  -a APPENDIX, --appendix APPENDIX
                        ''
```