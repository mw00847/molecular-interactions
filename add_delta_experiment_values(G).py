# load the combined dataset where the carbonyl related frequencies are 303-306
# extract 4 computed carbonyl peak frequencies per sample
# define 10 experimental peak frequencies
# computes frequency shifts (differences) between each computed and experimental peak
# reshapes the result from (499, 4, 10) → (499, 40)
# append 40 new features to the original dataset
# save the final dataset as .npy






import numpy as np

#load combined dataset
combined_data = np.load('directory! combined_dataset.npy', allow_pickle=True)
print("Loaded data shape:", combined_data.shape)

#extract computed carbonyl peak frequencies (Columns 303-306)
computed_frequencies = combined_data[:, 303:307]  # Shape: (499, 4)

#define experimental peak centers for 10 acetone fractions
experimental_peaks = np.array([
    1728.9990040673013, 1696.9439484537018, 1697.0947727914245, 1697.3760126912327,
    1697.4989848742132, 1698.5709617481805, 1699.0717186046627, 1700.1131163109806,
    1701.003928111602, 1701.8026388562123  # Using first 10 values for ML
])

#expand `experimental_peaks` across 499 rows** (499, 10)
experimental_peaks_expanded = np.tile(experimental_peaks, (computed_frequencies.shape[0], 1))

#verify shape consistency before subtraction
print(f"Computed frequencies shape: {computed_frequencies.shape}")  # (499, 4)
print(f"Expanded experimental peaks shape: {experimental_peaks_expanded.shape}")  # (499, 10)

#compute frequency shifts**
#compute the difference for EACH of the 4 computed frequencies against ALL 10 experimental peaks
#resulting shape: (499, 4, 10)
frequency_shifts = computed_frequencies[:, :, np.newaxis] - experimental_peaks_expanded[:, np.newaxis, :]

#reshape from (499, 4, 10) to (499, 40) so it can be concatenated
frequency_shifts = frequency_shifts.reshape(computed_frequencies.shape[0], -1)

#concatenate new frequency shift columns to `combined_data`
combined_data_with_target = np.concatenate([combined_data, frequency_shifts], axis=1)

#print final dataset shape
print("Final dataset shape with new frequency shift columns:", combined_data_with_target.shape)

#save the updated dataset
np.save('directory! combined_dataset_with_target.npy', combined_data_with_target)

print("dataset with 40 new target columns saved successfully!")