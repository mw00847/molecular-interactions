import numpy as np
import matplotlib.pyplot as plt

a = [-259.41544902, -266.45155711, -267.88991119, -268.27983544, -268.44914679,
     -268.57729483, -268.66750417, -268.72150953, -268.75080249, -268.76565194,
     -268.77281993, -268.77613025, -268.77753386, -268.77797257]

b = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]

plt.plot(b, a)
plt.xlabel('Variable b')
plt.ylabel('Variable a')
plt.title('Plot of a vs. b')
plt.grid(True)
plt.show()
