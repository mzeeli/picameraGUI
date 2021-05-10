"""
Houses the Cesium class. This is used as a central location to reference constants
relavent to the Cesium atom

Last Updated: Summer Term, 2021
Author: Michael Li
"""


class Cesium:
    """
    Cesium atom class, contains scientific constants to the cesium atom such as
    mass, linewidth, saturation intensity, etc...

    Data can be verified at https://steck.us/alkalidata/cesiumnumbers.pdf
    """
    def __init__(self):
        # Atomic Mass [kg]
        self.atomicMass = 2.206948425e-25

        # Natural line width of Cesium D2 [Hz]
        self.lineWidth = 5.22e6

        # saturation intensity of Cesium D2 [mW/cm^2]
        self.I_sat = 1.654
