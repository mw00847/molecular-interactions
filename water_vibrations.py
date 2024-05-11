
import psi4
import numpy as np
from scipy.spatial.transform import Rotation as R

# Set memory and output file
psi4.set_memory('3 GB')
psi4.core.set_output_file('output.dat', False)

# Basis set and computational method
basis_set = 'cc-pVDZ'
method = 'scf/' + basis_set

# Define original coordinates
oxygen = np.array([0.000000, 0.000000, 0.000000])
hydrogen1 = np.array([0.758602, 0.000000, 0.504284])
hydrogen2 = np.array([0.758602, 0.000000, -0.504284])

#determine the original angle
v1 = hydrogen1 - oxygen
v2 = hydrogen2 - oxygen
original_angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
print(original_angle)

#name the angle
angle_changes = np.arange(0, 90, 5)  # Adjust angles from -10 to +10 degrees in 5 degree increments

# Initialize a dictionary to store IR spectra
ir_spectra = {}

for delta_angle in angle_changes:
    new_angle = original_angle + delta_angle
    rotation_axis = np.cross(v1, v2)
    rotation_axis /= np.linalg.norm(rotation_axis)
    rotation_degree = new_angle - original_angle
    rotation = R.from_rotvec(rotation_degree * np.pi / 180 * rotation_axis)
    new_hydrogen1 = rotation.apply(v1) + oxygen
    new_hydrogen2 = rotation.apply(v2) + oxygen

    # Define the water molecule with the new angle
    water = psi4.geometry(f"""
    O  {oxygen[0]}  {oxygen[1]}  {oxygen[2]}
    H  {new_hydrogen1[0]}  {new_hydrogen1[1]}  {new_hydrogen1[2]}
    H  {new_hydrogen2[0]}  {new_hydrogen2[1]}  {new_hydrogen2[2]}
    """)

    # Calculate vibrational frequencies (includes IR intensities)
    try:
        frequency_analysis = psi4.frequency(method, molecule=water, return_c1=True)
        if isinstance(frequency_analysis, float):
            print("Frequency analysis failed to return the expected object. Returned a float instead.")
        else:
            ir_intensities = frequency_analysis.get_array('IR_INTENSITIES')
            frequencies = frequency_analysis.frequencies().to_array()

            # Store frequencies and intensities
            ir_spectra[new_angle] = (frequencies, ir_intensities)
    except psi4.p4util.exceptions.PsiException as e:
        print(f"Frequency calculation failed for angle {new_angle} degrees. Error: {str(e)}")
        continue

# Print the IR spectra for each angle
for angle, (freqs, ints) in ir_spectra.items():
    print(f"Angle: {angle:.2f} degrees")
    for i, freq in enumerate(freqs):
        if ints[i] > 0:  # Only display active IR modes
            print(f"{freq:.2f} cm^-1: {ints[i]:.2f} km/mol")
    print("\n")

print(ir_spectra)