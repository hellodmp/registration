import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


def read_dicom(dir, series_id):
    reader = sitk.ImageSeriesReader()
    series_file_names = {}
    series_IDs = reader.GetGDCMSeriesIDs(dir)
    if len(series_IDs) == 0:
        return None
    image = sitk.ReadImage(reader.GetGDCMSeriesFileNames(dir, series_id), sitk.sitkFloat32)
    return image


def sitk_show(nda, title=None, margin=0.0, dpi=40):
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
    for k in range(0, nda.shape[2]):
        print "printing slice " + str(k)
        ax.imshow(np.squeeze(nda[:, :, k]),cmap ='gray', extent=extent, interpolation=None)
        plt.draw()
        plt.pause(0.3)
        #plt.waitforbuttonpress()


def resample(sourceimg, spacing, newSize, method=sitk.sitkLinear):
    newSize = np.array(newSize)

    factor = np.asarray(sourceimg.GetSpacing()) / np.array(spacing)
    factorSize = np.asarray(sourceimg.GetSize() * factor, dtype=float)
    resampledSize = np.max([factorSize, newSize], axis=0)
    resampledSize = resampledSize.astype(dtype=int)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sourceimg)
    resampler.SetOutputSpacing([spacing[0], spacing[1], spacing[2]])
    resampler.SetSize(resampledSize)
    resampler.SetInterpolator(method)
    imgResampled = resampler.Execute(sourceimg)

    oldSize = np.array(imgResampled.GetSize())
    imgStartPx = ((oldSize - newSize) / 2.0).astype(dtype=int)
    regionExtractor = sitk.RegionOfInterestImageFilter()
    regionExtractor.SetSize(list(newSize.astype(dtype=int)))
    regionExtractor.SetIndex(list(imgStartPx))
    imgResampledCropped = regionExtractor.Execute(imgResampled)

    return imgResampledCropped

def plot_values(registration_method):
    print "metric value:", registration_method.GetMetricValue()

