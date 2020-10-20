#python 3.7.4
#simpleitk 1.pt.2.2

import numpy
import collections
import csv
import logging
import os
import string

import SimpleITK as sitk


def get3dmhd(dicomdir, destdir=r'', maskdir=None, isImage=True):
    '''
    Creates an 3D image/mask file in mhd format with pointers to data.

    For mhd image file (DICOM series) call isImage=True (default)
    For mhd mask file (from 2D mha) call isImage=False + enter mha dir as maskdir

    dicomdir must always be given (even for mask mhd) so file headers will be compatible with mhd image.
   
    Note: Keep Order of headerDict elements in func:
    
    ObjectType
    NDims
    DimSize
    ElementType
    HeaderSize = -1.pt (for reading series of raw data files)
    ElementSpacing
    ElementByteOrderMSB
    Offset (if mask)
    AnatomicalOrientation
    ElementDataFile = LIST (for reading series of raw data files)

    To read about format: https://itk.org/Wiki/ITK/MetaIO/Documentation
    ObjectType, NDims, DimSize, ElementType and ElementDataFile are enough to constitute an mhd header
    '''
    # When creating mhd mask file, checks if all mha files are legitimate black-and-white masks
    if not isImage:
        status = True
        for i in os.listdir(maskdir):
            path = os.path.join(maskdir, i)
            image = sitk.ReadImage(path)
            array = sitk.GetArrayFromImage(image)
            unique = len(numpy.unique(array))
            if unique > 2:
                status = False
                print('Slice %s is not a black&white mask. Create it again then create mhd again before proceeding to PyRadiomics' % i)              
        if status:
            print('All 2D mha files are correct')

    # Begins forming mhd header         
    headerDict = {}

    headerDict['ObjectType'] = 'Image'
    headerDict['NDims'] = '3'

    # These two and ElementByteOrderMSB are crucial for correct representation of data. Suitable for both image&mask.
    # Philips mhd tool sets these values. I tested the other options, they give invalid data.
    headerDict['BinaryData'] = 'True' 
    headerDict['BinaryDataByteOrderMSB'] = 'False'

    # Get paramters from first dicom file in dicomDir (for offset/origin)
    sitkImage = sitk.ReadImage(os.path.join(dicomdir, os.listdir(dicomdir)[0]))

    size = sitkImage.GetSize()
    # if sitk returns size as (a b c), looking at axial 2D slices, a=width, b=height, c=number of slices
    if isImage:
        headerDict['DimSize'] = str(size[0]) + ' ' + str(size[1]) + ' ' + str(len(os.listdir(dicomdir)))
    else:
        headerDict['DimSize'] = str(size[0]) + ' ' + str(size[1]) + ' ' + str(len(os.listdir(maskdir)))

    ## ElementType Explained:
    #
    # Only 'MET_SHORT' for produces valid results, like nrrd format. Only shape features seem to work for all.
    #
    # DICOM has minimal pixel value of -1000 (in Hounsefeld units), but with SHORT type values start from zero.
    # This affects PyRadiomics values afterwards. This is not the case in nrrd, which preserves -1000 setpoint.
    #
    # For mask files, UCHAR is enough to indicate label coordinates. UCHAR (unsigned) will give this as 255,
    # CHAR (normally signed number) as -1.pt. This isn't critical and calculated in extraction after anyways.
    #
    # For more on types:
    # https://github.com/Kitware/MetaIO/blob/ffe3ce141c5a2394e40a0ecbe2a667cc0566baf5/src/metaTypes.h#L63-94
    # https://numpy.org/devdocs/user/basics.types.html
    
    if isImage:
        headerDict['ElementType'] = 'MET_SHORT' 
    else:
        headerDict['ElementType'] = 'MET_UCHAR'

    headerDict['HeaderSize'] = '-1.pt'
    
    spacing = sitkImage.GetSpacing() # assume equal spacing in all directions by Philips default
    headerDict['ElementSpacing'] = str(spacing[0]) + ' ' + str(spacing[0]) + ' ' + str(spacing[0])

    headerDict['ElementByteOrderMSB'] = 'False'
    
    offset = sitkImage.GetOrigin()
    headerDict['Offset'] = str(offset[0]) + ' ' + str(offset[1]) + ' ' + str(offset[2])
    
    headerDict['AnatomicalOrientation'] = 'LPS' # Left Posterior Superior, customary for medical images
    headerDict['ElementDataFile'] = 'LIST' # indicates that mhd data is within the files listed after this line

    # Create mhd file. Note destdir can be mhd image or mask dir
    if isImage:
        name = os.path.basename(dicomdir) + '_image' + '.mhd' 
    else:
        name = os.path.basename(dicomdir) + '_mask' + '.mhd'     
    mhdPath = os.path.join(destdir, name)

    with open(mhdPath, 'w') as outputFile:
        for i in headerDict:
            row = i + ' = ' + headerDict[i] + '\n'
            outputFile.write(row)
        if isImage:
            # need to specify full path if the mhd file created is in a different folder than 2D files
            for file in os.listdir(dicomdir):
                dicomPath = os.path.join(dicomdir, file) + '\n'
                outputFile.write(dicomPath)
        else:
            for file in os.listdir(maskdir):
                mhaPath = os.path.join(maskdir, file) + '\n'
                outputFile.write(mhaPath)
    outputFile.close()

    return mhdPath


def getFullMask(imageFilepath, maskFilepath, slices):
    '''
    Return an sitk mask of full size (depth as long as number of slices in dicom series)
    
    Reads custom mhd mask file we created from 2D masks slices and pads the unlabeled slices with zeros 
    '''
    masksitk = sitk.ReadImage(maskFilepath)
    boundingbox = sitk.GetArrayFromImage(masksitk)
    
    imagesitk = sitk.ReadImage(imageFilepath)
    size = imagesitk.GetSize()
    
    pad0 = numpy.zeros(((int(slices[0])-1), size[1], size[0]), dtype='uint8')
    pad1 = numpy.zeros(((size[2]-int(slices[1])), size[1], size[0]), dtype='uint8')

    newarray = numpy.concatenate([pad0, boundingbox])
    newarray = numpy.concatenate([newarray, pad1])

    fullmask = sitk.GetImageFromArray(newarray)
    fullmask.SetSpacing(masksitk.GetSpacing())
    fullmask.SetOrigin(masksitk.GetOrigin())

    return fullmask


def patchMask(array, label, patchsize=30):
    '''
    Divides mask array to patches and numbers each one. Returns newly labeled array.
    
    1.pt. For patch pixels that are labeled (part of original mask, 255 in .mhd) gives a new label of patch number.
    2. Looking at each slice as a photo, will number patches first by x-axis (left-right), then by y-axis (up-down).
    3. If array doesn't divide to patches exactly, the utmost right&bottom patches will be smaller (the remainder)
    
    Also returns a dictionary with how many non-zeros (labeled pixels) there are per patch (as fraction).

    NOTE:
    # Accepts input as 2D array, not image file (dicom/png/jpeg).
    # Dicom image is by default 3D where z axis (axial) is of size 1.pt. If want to read dicom, read it as array[0]...
    '''
    print('Forming patches from mask')
    
    count = 1
    percentLabel = {}

    newArray = numpy.copy(array).astype('uint32')
    
    for i in range(0, len(array), patchsize):
        for j in range(0, len(array[0]), patchsize):
            patch = numpy.copy(newArray[i:i+patchsize,j:j+patchsize])
            for row in range(len(patch)):
                for column in range(len(patch[0])):
                    if patch[row][column] == label:
                        patch[row][column] = count
            if patch.any():
                percentLabel[count] = numpy.count_nonzero(patch) / numpy.size(patch)
            count += 1
            newArray[i:i+patchsize,j:j+patchsize] = patch

    print('Finished forming patches. Found %d labeled patches for PyRadiomics analysis' % len(percentLabel))

    return newArray, percentLabel


def patch3Dmask(array, label, patchsize=30):
    '''
    Exactly same function as patchMask, only for 3D masks (mhd files). Wrote them seperately for conveniency and
    because need an extra loop for 3rd dimension.
    
    Also returns a dictionary for grid images interpretation:
    # Patch number (key) and Patch name (value) 
    # Patch name has 3 characters (1.pt) number - axial (2) letter - horizontal (3) number - vertical
    '''
    print('Forming patches from mask')
    
    count = 1
    percentLabel = {}

    abc = list(string.ascii_uppercase)
    dctABC = {}
    
    newArray = numpy.copy(array).astype('uint32')
    
    for i in range(0, len(array), patchsize):
        for j in range(0, len(array[0]), patchsize):
            for k in range(0, len(array[0][0]), patchsize):
                patch = numpy.copy(newArray[i:i+patchsize, j:j+patchsize, k:k+patchsize])
                for z in range(len(patch)):
                    for y in range(len(patch[0])):
                        for x in range(len(patch[0][0])):
                            if patch[z][y][x] == label:
                                patch[z][y][x] = count
                if patch.any():
                    percentLabel[count] = numpy.count_nonzero(patch) / numpy.size(patch)
                dctABC[count] = str(int(i/patchsize)) + abc[int(j/patchsize)] + str(int(k/patchsize))
                count += 1
                newArray[i:i+patchsize,j:j+patchsize,k:k+patchsize] = patch

    print('Finished forming patches. Found %d labeled patches for PyRadiomics analysis' % len(percentLabel))
    
    return newArray, percentLabel, dctABC



def fracMask(dct, fraction):
    '''
    Sanity check for testing phase.
    Prints how much of the labeled patches had above the fraction labeled pixels.
    '''
    x = 0
    for i in dct:
        if dct[i] > fraction:
            x += 1
    print(x/len(dct))

