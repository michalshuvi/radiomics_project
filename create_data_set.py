import SimpleITK as sitk
from skimage.util import view_as_blocks
import os
from os import listdir
from os.path import isfile, join
import re
import csv
import random
import torch
from torchvision import transforms
import numpy as np
import operator
from PIL import Image
from radiomics import featureextractor


'''
The path of the "raw" data: contains dicom files if is_data_edited=False in the create_dataset function,
and otherwise PNG files; as well as the masks.
'''
DATA_PATH = "miceData"

#The mice names / folder names
MICE_IDS = ["MB1333220714F000H00000000C000000", "MB1444220714F000H00000000C000000", "MB1589241214FC00H00004052C000000",
            "MB1752210615FC00H00000557C000000", "MB1858230615FC00H00001912C000000", "MB1858140715FC00H00001912C000000",
            "MB1583050215FC00H00004052C000000", "MB1598010315FC00H00004457C000000", "MB1695040215F000H00000000C000000",
            "MB1588050215FC00H00004052C000000", "MB1472110315FC00H00004052C000000", "MB1755210615FC00H00000557C000000",
            "MB1953010315FC00H00004457C000000", "MB1468010315FC00H00003912C000000", "MB1513090614F000H00000000C000000",
            "MB1363150215F000H00000000C000000", "MB1370270414F000H00000000C000000", "MB1408220714F000H00000000C000000",
             "MB1452010315FC00H00000188C000000", "MB1512290714F000H00000000C000000",
             "MB1539110315FC00H00001912C000000", "MB1747010315FC00H00000072C000000", "MB1317200714F000H00000000C000000",
            "MB1464210615FC00H00000519C000000", "MB1488290614F000H00000000C000000", "MB1701240614F000H00000000C000000",
            "MB1727080914F000H00000000C000000", "MB1790241214FC00H00003348C000000"]

'''
NOISY_SCANS = ["MB1312240614F000H00000000C000000", "MB1392140814F000H00000000C000000", "MB1396110913F000H00000000C000000",
            "MB1398110913F000H00000000C000000", "MB1858100815FC00H00001912C000000",
            "MB4022091016F000H00000000C000000", "MB4134091016F000H00000000C000000"]
'''
#default size to reshape the dicom files to
MAX_SIZE = 300

'''
Semi-iteration class in which in every "next" call, the next patch is returned;
and None is returned if there is no more patches.
'''

class Patch_iter:
    def __init__(self, img_arr, patch_size):
        self.max_rows = (MAX_SIZE // patch_size) * patch_size
        self.max_cols = (MAX_SIZE // patch_size) * patch_size
        img_arr = img_arr[:self.max_rows, :self.max_cols]
        self.patches_arr = view_as_blocks(img_arr, block_shape=(patch_size, patch_size))
        self.row = 0
        self.col = 0
        self.patch_size = patch_size
        self.previous_patch = None

    def next(self):
        if self.previous_patch is not None and \
                self.previous_patch[0] >= (self.max_rows//self.patch_size)-1 and \
                self.previous_patch[1] >= (self.max_cols//self.patch_size)-1:
            return None

        old_row = self.row
        old_col = self.col
        self.previous_patch = (old_row, old_col)
        if self.row >= (self.max_rows//self.patch_size)-1 and self.col >= (self.max_cols//self.patch_size)-1:
            return self.patches_arr[old_row][old_col]
        if self.col >= (self.max_cols//self.patch_size)-1:
            self.row += 1
            self.col = 0
        else:
            self.col += 1
        return self.patches_arr[old_row][old_col]

    def get_previous_patch_number(self):
        return self.previous_patch


'''
create_dataset takes a speficic mouse, load its raw data from inp_path, calculate the patches
that need to be saved due to supplied conditions, and saves them as dictionaries that contain
all the important information (for more about the information, see save_as_dict).
The supplied conditions are:
- patch_size - the size of the patch, or in 3d case the size of the patch ignoring the "slice dimension"
- s_per_patch - how many slices will be included in a patch - 1 in a 2d patch, and more otherwise.
  In other words, the actual size of the patches are s_per_patch X patch_size X patch_size
- tumor_percent - the tumor percentage that half of the saved patches will contain and half won't contain
- data_per_mouse - the max number of data to save
- to_shuffle - if we want to move over the slices in a random way, and therefore create a non-deterministic
  data, to shuffle=True. Otherwise, to shuffle=False.
- is_data_edited - if the raw data's images are in DICOM format or without resize, is_data_edited=False.
  If it is in PNG format (or a similar format) and after resize, is_data_edited=True
- to_resize - in case the raw data is in DICOM format (is_data_edited is False), and the user doesn't want
  to resize the images (into MAX_SIZE X MAX_SIZE) - to_resize = False. If resize is needed, to_resize=True
'''

def create_dataset(mouse_ident, inp_path=DATA_PATH, folder_name="dataForNet", data_per_mouse=600, patch_size=50,
                               s_per_patch=1, tumor_percent=0.3, to_shuffle=True, is_data_edited=True, to_resize=False):

    transform = get_transform(slices_num=s_per_patch)
    index_counter = 1

    mask_path = inp_path + "/" + mouse_ident + "/masks"
    tumor_slices = set([int(re.findall("\d+", f)[0]) for f in listdir(mask_path) if isfile(join(mask_path, f))])
    dcm_path = inp_path + "/" + mouse_ident + "/images"
    img_names = [f for f in listdir(dcm_path) if isfile(join(dcm_path, f))]
    img_names.sort()

    data_tumor_counter = 0
    data_no_tumor_counter = 0

    slices = [i for i in range(1, (len(img_names)//s_per_patch))]
    if to_shuffle:
        random.shuffle(slices)

    for i in slices:
        if data_tumor_counter >= data_per_mouse//2 and data_no_tumor_counter >= data_per_mouse//2:
            break

        rows = MAX_SIZE // patch_size
        cols = MAX_SIZE // patch_size
        all_patches = [[] for x in range(s_per_patch)]
        all_masks = [[] for x in range(s_per_patch)]
        tumor_percent_in_patch = [[0 for c in range(cols)] for r in range(rows)]

        count_no_mask = 0
        for j in range(s_per_patch):
            all_patches[j] = [[0 for c in range(cols)] for r in range(rows)]
            all_masks[j] = [[0 for c in range(cols)] for r in range(rows)]

            image_path = dcm_path + "/" + img_names[(i-1)*s_per_patch+j]
            if is_data_edited:
                img = sitk.ReadImage(image_path)
                img = sitk.GetArrayFromImage(img)
            else:
                img = sitk.ReadImage(image_path)
                img = sitk.IntensityWindowing(img, -1000, 1000, 0, 255)
                img = sitk.Cast(img, sitk.sitkUInt8)
                if to_resize:
                    img = downsamplePatient(img, reduced_dim=True)
                    img = sitk.GetArrayFromImage(img)
                else:
                    img = sitk.GetArrayFromImage(img)[0]

            if ((i - 1) * s_per_patch + j + 1) not in tumor_slices:
                current_mask = None
                count_no_mask += 1
            else:
                if is_data_edited:
                    current_mask = sitk.ReadImage(mask_path + "/Mask" + str((i - 1) * s_per_patch + j + 1) + ".png")
                else:
                    if to_resize:
                        current_mask = downsamplePatient(mask_path + "/Mask" + str((i - 1) * s_per_patch + j + 1) + ".mha",
                                                 reduced_dim=False)
                    else:
                        current_mask = sitk.ReadImage(mask_path + "/Mask" + str((i - 1) * s_per_patch + j + 1) + ".mha")

                current_mask = sitk.GetArrayFromImage(current_mask)

            store_img_and_mask_patches(img, all_patches, all_masks, patch_size,
                                       j, s_per_patch, tumor_percent_in_patch, current_mask)

        for r in range(rows):
            for c in range(cols):
                cond_for_tumor = (tumor_percent_in_patch[r][c] >= tumor_percent and
                                  data_tumor_counter < data_per_mouse//2)
                cond_for_no_tumor = (tumor_percent_in_patch[r][c] < tumor_percent
                                     and data_no_tumor_counter < data_tumor_counter)

                if cond_for_tumor or cond_for_no_tumor:
                    save_as_dict(transform, all_patches, all_masks, tumor_percent_in_patch, mouse_ident,
                                 folder_name, index_counter, s_per_patch, i, r, c)
                    index_counter += 1
                    if cond_for_tumor:
                        data_tumor_counter += 1
                    else:
                        data_no_tumor_counter += 1



'''
count_occurrences takes a 2d square matrix and returns the number of pixels that
are big/small/equal to a specific entry (decided by the operator that was supplied).
If to_change=True every pixel that meets the condition is changed to 255, and 
every pixel that doesn't meet the condition is changed to 0
'''

def count_occurrences(arr, entry, size, oper='=', ratio=1, to_change=False):
    count = 0
    for r in range(size):
        for c in range(size):
            if get_truth(arr[r][c], oper, entry*ratio):
                count += 1
                if to_change:
                    arr[r][c] = 255
            else:
                if to_change:
                    arr[r][c] = 0
    return count


'''
get_truth returns the truth value determined by two numbers and an operator
'''


def get_truth(inp, relate, cut):
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '=': operator.eq}
    return ops[relate](inp, cut)


"""
save_as_dict saves a patch (can be 3d) as a dictionary contains all the important information,
such as the patch's image, the patch's mask, the slice_range and more.
"""


def save_as_dict(transform, all_patches, all_masks, tumor_percent_in_patch, mouse_ident,
                 folder_name, index_counter, s_per_patch, i, r, c):
    patch_dict = dict()
    patch_dict["patch"] = (r, c)
    patch_dict["mouse_name"] = mouse_ident
    patch_dict["slice_range"] = (((i - 1) * s_per_patch + 1), (i * s_per_patch))
    patch_dict["tumor_percentage"] = tumor_percent_in_patch[r][c]

    out_3d_img = [convert_to_255(all_patches[0][r][c])]
    out_3d_mask = [all_masks[0][r][c]]
    for s in range(1, s_per_patch):
        tmp1 = convert_to_255(all_patches[s][r][c])
        out_3d_img.append(tmp1)
        out_3d_mask.append(all_masks[s][r][c])

    out_3d_img = np.stack(out_3d_img, axis=2)
    out_3d_mask = np.array(out_3d_mask)

    patch_dict["image"] = transform(out_3d_img)
    patch_dict["mask"] = out_3d_mask

    file_out = folder_name + "/" + mouse_ident + "/"
    torch.save(patch_dict, file_out + str(index_counter))


"""
store_img_and_mask_patches stores patches, masks and tumor percentages
"""


def store_img_and_mask_patches(img, all_patches, all_masks, patch_size,
                               j, s_per_patch, tumor_percent_in_patch, current_mask):
    patch_mask = None
    patch_img_iter = Patch_iter(img, patch_size)
    patch_img = patch_img_iter.next()

    if current_mask is not None:
        patch_mask_iter = Patch_iter(current_mask, patch_size)
        patch_mask = patch_mask_iter.next()

    while patch_img is not None:
        if patch_mask is not None:
            occ = count_occurrences(arr=patch_mask, entry=255, size=patch_size,
                                    oper=">", ratio=0.5, to_change=True)
            percent = occ / (patch_size * patch_size)
            temp = patch_mask
        else:
            percent = 0
            temp = [[0 for c in range(patch_size)] for r in range(patch_size)]

        p_n = patch_img_iter.get_previous_patch_number()

        all_patches[j][p_n[0]][p_n[1]] = patch_img
        all_masks[j][p_n[0]][p_n[1]] = temp
        tumor_percent_in_patch[p_n[0]][p_n[1]] += (percent / s_per_patch)

        patch_img = patch_img_iter.next()
        if patch_mask is not None:
            patch_mask = patch_mask_iter.next()


'''
convert_to_255 takes an 2d array, normalize it and multiply by 255 to change the range to 0-255
'''


def convert_to_255(array):
    new_array = np.array(array)
    if new_array.max() == 0 and new_array.min() == 0:
        return new_array
    new_array = (255.0 / new_array.max() * (new_array - new_array.min())).astype(np.uint8)
    return new_array


'''
get_transform returns a function that takes a numpy array and returns a normalized matching tensor
'''


def get_transform(slices_num=30):
    values = [transforms.ToTensor(),
              transforms.Normalize((0.5,) * slices_num, (0.5,) * slices_num)]
    return transforms.Compose(values)


'''
downsamplePatient takes a 2d sitk image and returns a resized MAX_SIZE X MAX_SIZE image.
According to SimpleITK's documentation, the process of image resampling involves 4 steps:
Image - the image we resample, given in the coordinate system;
Resampling grid - a regular grid of points given in a coordinate system which will be mapped to the coordinate system;
Transformation - maps points from the coordinate system to coordinate system;
Interpolator - a method for obtaining the intensity values at arbitrary points in the coordinate system from the values of the points defined by the Image
'''

def downsamplePatient(patient_CT, reduced_dim=True):
    if isinstance(patient_CT, str):
        original_CT = sitk.ReadImage(patient_CT, sitk.sitkInt32)
    else:
        original_CT = patient_CT

    if reduced_dim:
        original_CT = original_CT[:, :, 0]
    dimension = original_CT.GetDimension()
    reference_physical_size = np.zeros(original_CT.GetDimension())
    reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                  zip(original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size)]

    reference_origin = original_CT.GetOrigin()
    reference_direction = original_CT.GetDirection()

    reference_size = [MAX_SIZE for sz in original_CT.GetSize()]
    reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, original_CT.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(original_CT.GetDirection())

    transform.SetTranslation(np.array(original_CT.GetOrigin()) - reference_origin)

    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(original_CT.TransformContinuousIndexToPhysicalPoint(np.array(original_CT.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)

    #sitk.Show(sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0))
    return sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0)


'''
"main" calls create_dataset on every mouse from a specific list, or MICE_IDS by deafult, 
which contains all of the 29 mice
'''


def main(mice_names=MICE_IDS, inp_path=DATA_PATH, data_per_mouse=600, patch_size=50, s_per_patch=1, tumor_percent=0.3,
         folder_name="dataForNet", to_shuffle=True, is_data_edited=True, to_resize=False):
    for m in mice_names:
        if not os.path.isdir(folder_name + "/" + m):
            os.makedirs(folder_name + "/" + m)
        print("started", m)
        create_dataset(inp_path=inp_path, mouse_ident=m, data_per_mouse=data_per_mouse, patch_size=patch_size,
                                   s_per_patch=s_per_patch, tumor_percent=tumor_percent,
                                   folder_name=folder_name, is_data_edited=is_data_edited,
                                   to_shuffle=to_shuffle, to_resize=to_resize)
        print("finished", m)
