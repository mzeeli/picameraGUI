"""
Cesium atom class, contains relavent information to the cesium atom such as
mass, linewidth, saturation intensity, etc...

Last Updated: Winter, 2021
Author: Michael Li
"""


class Cesium:
    def __init__(self):
        # Atomic Mass [kg]
        self.atomicMass = 2.206948425e-25

        # Natural line width of Cesium D2 [Hz]
        self.lineWidth = 5.22e6

        # saturation intensity of Cesium D2 [mW/cm^2]
        self.I_sat = 1.654
