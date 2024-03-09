from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.composition.element import Stoichiometry,TMetalFraction
from matminer.featurizers.composition.ion import OxidationStates,IonProperty,ElectronAffinity,ElectronegativityDiff
from matminer.featurizers.conversions import CompositionToOxidComposition
from matminer.featurizers.site.bonding import BondOrientationalParameter,AverageBondLength
from pymatgen.analysis.local_env import NearNeighbors
from matminer.featurizers.structure.order import DensityFeatures
from matminer.featurizers.structure.sites import SiteStatsFingerprint

class element_property():
    def __init__(self,compositys):
        self.composity = compositys

    def get_feature(self):
        f = []
        # stats = ["minimum", "maximum", "range", "mean", "std_dev"]
        stats = ["mean"]
        features = [
            "Polarizability", # Static average electric dipole polarizability
            #"n_ws^third", # Electron density at surface of Wigner-Sietz cell. Used in Miedema's model
            "GSvolume_pa", # DFT volume per atom of T=0K ground state
            "FirstIonizationEnergy", # Energy to remove the first electron from an element
            "Density", # Density of element at STP
            "Number", # Atomic number
            "MendeleevNumber", # Mendeleev Number
            "CovalentRadius", # covalent radius
            "NsValence",
            "NpValence",
            "NdValence", # Number of filled d valence orbitals
            "NfValence",
            "NValence", # Number of valence electrons
            "NsUnfilled",
            "NpUnfilled",
            "NdUnfilled", # Number of unfilled d valence orbitals
            "NfUnfilled",
            "NUnfilled", # Number of unfilled valence orbitals
        ]
        for i in ElementProperty("magpie", features, stats).featurize(self.composity):
            f.append(i)

        stats = ["mean"]
        features = [
            "X",
            "atomic_mass",
            "atomic_radius",
        ]
        for i in ElementProperty("pymatgen", features, stats).featurize(self.composity):
            f.append(i)

        stats = ["mean"]
        features = [
            "row_num",
            "col_num",
            "atom_radius",
            "molar_vol",
        ]
        for i in ElementProperty("deml", features, stats).featurize(self.composity):
            f.append(i)
        return f

    def get_lable(self):
        f = []
        stats = ["mean"]
        features = [
            "Polarizability",  # Static average electric dipole polarizability
            # "n_ws^third", # Electron density at surface of Wigner-Sietz cell. Used in Miedema's model
            "GSvolume_pa",  # DFT volume per atom of T=0K ground state
            "FirstIonizationEnergy",  # Energy to remove the first electron from an element
            "Density",  # Density of element at STP
            "Number",  # Atomic number
            "MendeleevNumber",  # Mendeleev Number
            "CovalentRadius",  # covalent radius
            "NsValence",
            "NpValence",
            "NdValence",  # Number of filled d valence orbitals
            "NfValence",
            "NValence",  # Number of valence electrons
            "NsUnfilled",
            "NpUnfilled",
            "NdUnfilled",  # Number of unfilled d valence orbitals
            "NfUnfilled",
            "NUnfilled",  # Number of unfilled valence orbitals
        ]
        for i in ElementProperty("magpie", features, stats).feature_labels():
            f.append(i)

        stats = ["mean"]
        features = [
            "X",
            "atomic_mass",
            "atomic_radius",
        ]
        for i in ElementProperty("pymatgen", features, stats).feature_labels():
            f.append(i)

        stats = ["mean"]
        features = [
            "row_num",
            "col_num",
            "atom_radius",
            "molar_vol",
        ]
        for i in ElementProperty("deml", features, stats).feature_labels():
            f.append(i)
        return f


    def get_only_feature(self):
        f = []
        # stats = ["minimum", "maximum", "range", "mean", "std_dev"]
        stats = ["mean"]
        features = [
            "Polarizability", # Static average electric dipole polarizability
            "n_ws^third", # Electron density at surface of Wigner-Sietz cell. Used in Miedema's model
            "GSvolume_pa", # DFT volume per atom of T=0K ground state
            "FirstIonizationEnergy", # Energy to remove the first electron from an element
            "Density", # Density of element at STP
            "Number", # Atomic number
            "MendeleevNumber", # Mendeleev Number
            "CovalentRadius", # covalent radius
            "NsValence",
            "NpValence",
            "NdValence", # Number of filled d valence orbitals
            "NfValence",
            "NValence", # Number of valence electrons
            "NsUnfilled",
            "NpUnfilled",
            "NdUnfilled", # Number of unfilled d valence orbitals
            "NfUnfilled",
            "NUnfilled", # Number of unfilled valence orbitals
        ]
        for i in ElementProperty("magpie", features, stats).featurize(self.composity):
            f.append(i)

        stats = ["mean"]
        features = [
            "X",
            "atomic_mass",
            "atomic_radius",
        ]
        for i in ElementProperty("pymatgen", features, stats).featurize(self.composity):
            f.append(i)

        stats = ["mean"]
        features = [
            "row_num",
            "col_num",
            "atom_radius",
            "molar_vol",
        ]
        for i in ElementProperty("deml", features, stats).featurize(self.composity):
            f.append(i)
        return f

    def get_only_lable(self):
        f = []
        stats = ["mean"]
        features = [
            "Polarizability",  # Static average electric dipole polarizability
            "n_ws^third",  # Electron density at surface of Wigner-Sietz cell. Used in Miedema's model
            "GSvolume_pa",  # DFT volume per atom of T=0K ground state
            "FirstIonizationEnergy",  # Energy to remove the first electron from an element
            "Density",  # Density of element at STP
            "Number",  # Atomic number
            "MendeleevNumber",  # Mendeleev Number
            "CovalentRadius",  # covalent radius
            "NsValence",
            "NpValence",
            "NdValence",  # Number of filled d valence orbitals
            "NfValence",
            "NValence",  # Number of valence electrons
            "NsUnfilled",
            "NpUnfilled",
            "NdUnfilled",  # Number of unfilled d valence orbitals
            "NfUnfilled",
            "NUnfilled",  # Number of unfilled valence orbitals
        ]
        for i in ElementProperty("magpie", features, stats).feature_labels():
            f.append(i)

        stats = ["mean"]
        features = [
            "X",
            "atomic_mass",
            "atomic_radius",
        ]
        for i in ElementProperty("pymatgen", features, stats).feature_labels():
            f.append(i)

        stats = ["mean"]
        features = [
            "row_num",
            "col_num",
            "atom_radius",
            "molar_vol",
        ]
        for i in ElementProperty("deml", features, stats).feature_labels():
            f.append(i)
        return f