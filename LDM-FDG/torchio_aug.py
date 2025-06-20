#%% Import libraries and setup directories
import numpy as np 
import random 
import torchio as tio

#%% Define the Augmentations class
class Augmentations:
    def __init__(self):
        self.random_anisotropy = tio.RandomAnisotropy(axes=(0, 1))
        self.random_affine = tio.RandomAffine()
        self.add_motion = tio.RandomMotion(num_transforms=1, image_interpolation='nearest')
        self.rescale = tio.RescaleIntensity((0, 1))

    def __call__(self, subject):
        aug_level = random.randint(1, 3)
        # Define individual transformations
        def blur(subject):
            downsampling_factor = random.randint(2, 3)
            original_spacing = 1 # This might need to be adjusted based on actual pixel spacing
            std = tio.Resample.get_sigma(downsampling_factor, original_spacing)
            antialiasing = tio.Blur(std) # Axes will default. Check if it's implicitly handling 2D correctly.
            return antialiasing(subject)

        def anistropy(subject):
            return self.random_anisotropy(subject)

        def affine(subject):
            return self.random_affine(subject)

        def elastix(subject):
            max_displacement_value = random.randint(1, 5) # Still in voxels
            # For 2D images, the axes should be (0, 1)
            random_elastic = tio.RandomElasticDeformation(
                max_displacement=max_displacement_value,
                num_control_points=random.randint(5, 15),
            )
            return random_elastic(subject)

        def noise(subject):
            add_noise = tio.RandomNoise(std=(np.random.rand() / 4))
            return add_noise(subject)

        def field_bias(subject):
            add_bias = tio.RandomBiasField(coefficients=(np.random.rand() / 2))
            return add_bias(subject)

        def motion(subject):
            return self.add_motion(subject)

        # List of functions
        all_functions = [blur, anistropy, noise, field_bias, motion]
        blur_functions = [noise, field_bias, anistropy]
        other_functions = [motion, blur]

        # Select transformations based on augmentation level
        if aug_level == 0:
            selected_functions = []
        elif aug_level == 1:
            selected_functions = random.sample(all_functions, 1)
        elif aug_level == 2:
            selected_blur_functions = random.sample(blur_functions, 1)
            selected_other_functions = random.sample(other_functions, 1)
            selected_functions = selected_blur_functions + selected_other_functions
        elif aug_level == 3:
            selected_blur_functions = random.sample(blur_functions, 2)
            selected_other_functions = random.sample(other_functions, 2)
            selected_functions = selected_blur_functions + selected_other_functions

        # Apply transformations
        subject = affine(subject)
        subject = elastix(subject)

        for func in selected_functions:
            subject = func(subject)

        return self.rescale(subject)
