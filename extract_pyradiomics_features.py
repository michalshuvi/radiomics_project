import csv
import os
import SimpleITK as sitk


def main():
    for mouse in mice:
        outputFilepath = os.path.join(outPath, 'radiomics_features.csv')
        sitkMask = getFullMask(imageFilepath, maskFilepath, slices)
        sitkarray = sitk.GetArrayFromImage(sitkMask)
        patchedMask, dctLabel, dctABC = patch3Dmask(sitkarray, totalLabel, patchsize=patchsize)
        newMask = sitk.GetImageFromArray(patchedMask)
        newMask.CopyInformation(sitkMask)
        for label in dctLabel:
            feature_vector = collections.OrderedDict(entry)
            feature_vector['Image'] = os.path.basename(imageFilepath)
            feature_vector['Mask'] = os.path.basename(maskFilepath)
            feature_vector['PatchNumber'] = label
            feature_vector['PatchName'] = dctABC[label]
            feature_vector['FractionLabelInPatch'] = dctLabel[label]
            try:
                feature_vector.update(extractor.execute(imageFilepath, newMask, label, voxelBased=True))
                writeResults(feature_vector, outputFilepath, count)
                count += 1
            except Exception:
                logger.error('FEATURE EXTRACTION FAILED', exc_info=True)
                writeResults(feature_vector, outputFilepath, count)


def writeResults(feature_vector, outputFilepath, count):
    '''
    Creates excel file with results from feature extraction
    '''
    with open(outputFilepath, 'a') as outputFile:
        writer = csv.writer(outputFile, lineterminator='\n')
        # assumes first run of feature extraction will succeed, otherwise won't write features headers for next rows
        headers = feature_vector.keys()
        if count == 0:
            headers = list(headers)
            writer.writerow(headers)

        row = []
        for h in headers:
            row.append(feature_vector.get(h, "N/A"))
        writer.writerow(row)
