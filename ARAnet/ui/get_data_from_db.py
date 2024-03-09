import numpy as np
from pymatgen.electronic_structure.bandstructure import BandStructure
import json

f = open('./data/number.CSV')
data_ = f.readlines()
f.close()
number_p = {}
for i in data_:
    a = i.split()[0].split(',')
    number_p[int(a[0])] = a[1]


class main_info():
    def __init__(self, Row):
        self.Row = Row

    def cell(self):
        cell = self.Row["cell"]
        return cell

    def energy(self):
        energy = self.Row["energy"]
        return energy

    def forces(self):
        forces = self.Row["forces"]
        return forces

    def free_energy(self):
        free_energy = self.Row["free_energy"]
        return free_energy

    def initial_magmoms(self):
        initial_magmoms = self.Row["initial_magmoms"]
        return initial_magmoms

    def magmom(self):
        magmom = self.Row["magmom"]
        return magmom

    def magmoms(self):
        magmoms = self.Row["magmoms"]
        return magmoms

    def atom_numbers(self):
        numbers = self.Row["numbers"]
        numbers = np.sort(numbers)
        element = []
        for i in numbers:
            element.append(str(number_p[i]))
        return [numbers, element]

    def pbc(self):
        pbc = self.Row["pbc"]
        return pbc

    def positions(self):
        positions = self.Row["positions"]
        return positions

    def stress(self):
        stress = self.Row["stress"]
        return stress

    def unique_id(self):
        unique_id = self.Row["unique_id"]
        return unique_id


class structure():
    def __init__(self, Row):
        self.data = Row["data"]["structure.json"]["1"]
        self.data_info = Row["data"]["results-asr.structureinfo.json"]["kwargs"]["data"]

    def get_formula(self):
        # formula
        formula = self.data_info["formula"]
        return formula

    def get_area_of_unit_cell(self):
        # area of unit cell (A**2)
        area_of_unit_cell = self.data_info["cell_area"]
        return area_of_unit_cell

    def get_has_inversion_symmetry(self):
        # material has inversion symmetry?
        inversion_symmetry = self.data_info["has_inversion_symmetry"]
        return inversion_symmetry

    def get_stoichiometry(self):
        # stoichiometry
        stoichiometry = self.data_info["stoichiometry"]
        return stoichiometry

    def get_spacegroup(self):
        # spacegroup
        spacegroup = self.data_info["spacegroup"]
        return spacegroup

    def get_space_group_number(self):
        # Spacegroup number
        spacegroup_number = self.data_info["spgnum"]
        return spacegroup_number

    def get_layergroup(self):
        # spacegroup
        layergroup = self.data_info["layergroup"]
        return layergroup

    def get_layer_group_number(self):
        # Spacegroup number
        layergroup_number = self.data_info["lggnum"]
        return layergroup_number

    def get_point_group(self):
        # Point group
        pointgroup = self.data_info["pointgroup"]
        return pointgroup

    def get_crystal_type(self):
        # Crystal type
        crystal_type = self.data_info["crystal_type"]
        return crystal_type

    def cell(self):
        cell = self.data["cell"]
        return np.array(cell)

    def dipole(self):
        dipole = self.data["dipole"]
        return dipole

    def energy(self):
        energy = self.data["energy"]
        return energy

    def forces(self):
        forces = self.data["forces"]
        return forces

    def free_energy(self):
        free_energy = self.data["free_energy"]
        return free_energy

    def initial_magmoms(self):
        initial_magmoms = self.data["initial_magmoms"]
        return initial_magmoms

    def magmom(self):
        magmom = self.data["magmom"]
        return magmom

    def magmoms(self):
        magmoms = self.data["magmoms"]
        return magmoms

    def atom_numbers(self):
        numbers = self.data["numbers"]
        element = []
        for i in numbers:
            element.append(str(number_p[i]))
        return [numbers, element]

    def pbc(self):
        pbc = self.data["pbc"]
        return pbc

    def positions(self):
        positions = self.data["positions"]
        return positions

    def stress(self):
        stress = self.data["stress"]
        return stress

    def tags(self):
        try:
            tags =self.data["tags"]
        except:
            tags = None
        return tags

    def Write_POSCAR(self, filename):
        cell = structure.cell(self)
        positions_number = structure.atom_numbers(self)[0]
        print(positions_number)
        positions = structure.positions(self)
        print(positions)
        formula = structure.get_formula(self)
        p = {}
        for i in positions_number:
            if number_p[int(i)] not in p:
                p[number_p[int(i)]] = 1
            else:
                p[number_p[int(i)]] += 1
        keys = p.keys()
        values = p.values()

        f = {}
        for i in range(len(positions_number)):
            if number_p[int(positions_number[i])] not in f:
                f[number_p[int(positions_number[i])]] = [positions[i]]
            else:
                f[number_p[int(positions_number[i])]].append(positions[i])

        with open(filename, "w") as fp:
            fp.write("{}\n".format(formula))
            fp.write("1.0\n")
            for i in range(len(cell)):
                fp.write("     %-25s\t%-25s\t%-25s\n" % (cell[i][0], cell[i][1], cell[i][2]))
            for i in keys:
                fp.write("    {}".format(i))
            fp.write("\n")
            for i in values:
                fp.write("    {}".format(i))
            fp.write("\n")
            fp.write("Cartesian\n")

            for i in f.keys():
                for j in f[i]:
                    fp.write("     %-25s\t%-25s\t%-25s\t%-25s\n" % (j[0], j[1], j[2], i))


class PBE_band():
    def __init__(self, Row):
        self.data = Row["data"]["results-asr.gs.json"]["kwargs"]["data"]

    def get_gaps(self):
        try:
            return self.data["gaps_nosoc"]["kwargs"]["data"]
        except:
            return None

    def vacuum_levels(self):
        try:
            return self.data["vacuumlevels"]["kwargs"]["data"]
        except:
            return None

class HSE_band():
    def __init__(self, Row):
        self.data = Row["data"]["results-asr.hse.json"]["kwargs"]["data"]

    def vbm_hse(self):
        try:
            return self.data["vbm_hse_nosoc"]
        except:
            return None

    def cbm_hse(self):
        try:
            return self.data["cbm_hse_nosoc"]
        except:
            return None

    def gap_dir(self):
        try:
            return self.data["gap_dir_hse_nosoc"]
        except:
            return None

    def gap_hse(self):
        try:
            return self.data["gap_hse_nosoc"]
        except:
            return None

    def kvbm(self):
        try:
            return self.data["kvbm_nosoc"]
        except:
            return None

    def kcbm(self):
        try:
            return self.data["kcbm_nosoc"]
        except:
            return None

    def vbm_hse_soc(self):
        return self.data["vbm_hse"]

    def cbm_hse_soc(self):
        return self.data["cbm_hse"]

    def gap_dir_soc(self):
        return self.data["gap_dir_hse"]

    def gap_hse_soc(self):
        return self.data["gap_hse"]

    def kvbm_soc(self):
        return self.data["kvbm"]

    def kcbm_soc(self):
        return self.data["kcbm"]

    def efermi_hse(self):
        return self.data["efermi_hse_nosoc"]

    def efermi_hse_soc(self):
        return self.data["efermi_hse_soc"]

    def cbm_dir_hse(self):
        try:
            return self.data["cbm_hse_nosoc"]
        except:
            return None

    def vbm_dir_hse(self):
        try:
            cbm = self.data["cbm_hse_nosoc"]
        except:
            cbm = None
        try:
            vbm = self.data["vbm_hse_nosoc"]
        except:
            vbm = None
        try:
            dir = self.data["gap_dir_hse_nosoc"]
        except:
            dir = None

        if cbm != None and vbm != None and dir != None:
            return cbm-dir
        else:
            return None
