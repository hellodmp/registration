import SimpleITK as sitk
from ipywidgets import interact, fixed
from  utility import *

def show(image):
    image1 = np.transpose(sitk.GetArrayFromImage(image).astype(dtype=float), [1, 2, 0])
    sitk_show(image1)
    print image1.shape

def show_with_alpha(moving_image, fixed_image, alpha):
    moving_image = np.transpose(sitk.GetArrayFromImage(moving_image).astype(dtype=float), [1, 2, 0])
    fixed_image = np.transpose(sitk.GetArrayFromImage(fixed_image).astype(dtype=float), [1, 2, 0])
    img = (1.0 - alpha)*moving_image + alpha*fixed_image
    sitk_show(img)
    print img.shape

def registration(fixed_image, moving_image):
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    #registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    #registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    #registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))
    return final_transform


def centre_registraion(fixed_image, moving_image):
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    moving_resampled = sitk.Resample(moving_image, fixed_image,
                                     initial_transform, sitk.sitkLinear,
                                     0.0, moving_image.GetPixelID())
    return moving_resampled


def test1():
    fixed_image = sitk.ReadImage("./data/training_001_ct.mha", sitk.sitkFloat32)
    moving_image = sitk.ReadImage("./data/training_001_mr_T1.mha", sitk.sitkFloat32)

    print "fixed=", fixed_image.GetSpacing(), fixed_image.GetSize()
    print "moving=", moving_image.GetSpacing(),  moving_image.GetSize()

    '''
    final_transform = registration(fixed_image, moving_image)
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())
    '''

    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    moving_resampled = sitk.Resample(moving_image, fixed_image,
                                     initial_transform, sitk.sitkLinear,
                                     0.0, moving_image.GetPixelID())

    show(moving_resampled)

    #show_with_alpha(fixed_image, moving_resampled, 0.7)
    print "end"


def test2():
    data_directory = "./data/V20940/"
    ct_id = "1.3.12.2.1107.5.1.4.49611.30000016090706002395300002514"
    mr_id = "1.3.12.2.1107.5.2.19.45362.2016090909281031592921966.0.0.0"

    fixed_image = read_dicom(data_directory,ct_id)
    moving_image = read_dicom(data_directory, mr_id)
    spacing=[1.484375, 1.484375, 5.0]
    fixed_image = resample(fixed_image, spacing, (256, 256, 93), method=sitk.sitkLinear)
    moving_image = resample(moving_image, spacing, (256, 256, 15), method=sitk.sitkLinear)
    print "fixed=", fixed_image.GetSpacing(), fixed_image.GetSize()
    print "moving=", moving_image.GetSpacing(),  moving_image.GetSize()
    #show(fixed_image)
    '''
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    moving_resampled = sitk.Resample(fixed_image,moving_image,
                                     initial_transform, sitk.sitkLinear,
                                     0.0, moving_image.GetPixelID())
    #show_with_alpha(fixed_image, moving_resampled, 0.7)
    show(moving_resampled)
    '''
    final_transform = registration(fixed_image, moving_image)
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())
    #show(moving_resampled)
    show_with_alpha(fixed_image, moving_resampled, 0.7)
    print "end"




if __name__ == "__main__":
    test1()
    #test2()

