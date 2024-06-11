#check imaginary frequencies in the GAMESS output

import sys
import os



#loop through all the .log files in the directory and check for imaginary frequencies
for file in os.listdir('.'):
    if file.endswith('.log'):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'FREQUENCY:' in line:
                    if 'I' in line:
                        print('Imaginary frequency found in file: ' + file)
                        print('Line number: ' + str(i))
                        print('Line: ' + line)
                        print('\n')
                    else:
                        continue
                else:
                    continue
    else:
        continue
        