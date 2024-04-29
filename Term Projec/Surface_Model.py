"""
Topographic Modeling including surface generation with roughness and crater distribution

"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import fftpack
import Surface_Model as SM_lib

@dataclasses.dataclass
class Data:
    """
    structure containing data for map

    :param height: terrain height
    """
    height: np.array = None
    color: np.array = None

@dataclasses.dataclass
class property_crater:
    """
    structure containing crater property

    :param distribution: crater distribution ("random" and "single")
    :param geometry: geometry type from "normal", "mound", "flat", and "concentric"
    :param min_D: minimum crater range
    :param max_D: maximum crater range
    :param con_D: constant crater range
    """
    distribution: str = None
    geometry: str = None
    min_D: float = None
    max_D: float = None
    con_D: float = None

@dataclasses.dataclass
class parameters:
    """
    structure containing parameters for map

    :param n: # of grid in one axis
    :param res: grid resolution [m]
    :param re: roughness exponent for fractal surface (0 < re < 1)
    :param sigma: amplitude gain for fractal surface
    :param is_fractal: choose to apply fractal surface
    :param is_crater: choose to apply crater shape
    :param crater_prop: crater property denoted as "CraterProp" data structure
    """
    n: int = None
    res: float = None
    re: float = None
    sigma: float = None
    is_fractal: bool = None
    is_crater: bool = None
    crater_prop: any = None
    
    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class grid_map:

    def __init__(surf_m, param, seed: int = 0):
       
        """
         grid size (param.n)
         resolution (param.res)
         center position (surf_m.c_x and surf_m.c_y), 
         lower left position (surf_m.lower_left_x and surf_m.lower_left_y),
         data array (surf_m.data).

        :param param: structure containing map parameters
        :param seed: random seed 
        """
        # set given parameters
        surf_m.param = param
        # identify center and lower left positions
        surf_m.c_x = surf_m.param.n * surf_m.param.res / 2.0
        surf_m.c_y = surf_m.param.n * surf_m.param.res / 2.0
        surf_m.lower_left_x = surf_m.c_x - surf_m.param.n / 2.0 * surf_m.param.res
        surf_m.lower_left_y = surf_m.c_y - surf_m.param.n / 2.0 * surf_m.param.res

        # generate data array
        surf_m.param.num_grid = surf_m.param.n**2
        surf_m.data = Data(height=np.zeros(surf_m.param.num_grid))

        # set randomness
        surf_m.seed = seed
        surf_m.set_randomness()

    def set_randomness(surf_m):
        """
        set_randomness: set randomness for reproductivity

        """
        if surf_m.seed is not None:
            surf_m.rng = np.random.default_rng(surf_m.seed)
        else:
            surf_m.rng = np.random.default_rng()

    # following functions are used for basic operations
    def get_value_from_xy_id(surf_m, x_id: int, y_id: int, field_name: str = "height"):
        """
        get_value_from_xy_id: get values at specified location described as x- and y-axis indices from data structure

        :param x_id: x index
        :param y_id: y index
        :param field_name: structure's name
        """
        grid_id = surf_m.calc_grid_id_from_xy_id(x_id, y_id)

        if 0 <= grid_id < surf_m.param.num_grid:
            data = getattr(surf_m.data, field_name)
            return data[grid_id]
        else:
            return None

    def get_xy_id_from_xy_pos(surf_m, x_pos: float, y_pos: float):
        """
        get_xy_id_from_xy_pos: get x- and y-axis indices for given positional information

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        """
        x_id = surf_m.calc_xy_id_from_pos(x_pos, surf_m.lower_left_x, surf_m.param.n)
        y_id = surf_m.calc_xy_id_from_pos(y_pos, surf_m.lower_left_y, surf_m.param.n)

        return x_id, y_id

    def calc_grid_id_from_xy_id(surf_m, x_id: int, y_id: int):
        """
        calc_grid_id_from_xy_id: calculate one-dimensional grid index from x- and y-axis indices (2D -> 1D transformation)

        :param x_id: x index
        :param y_id: y index
        """
        grid_id = int(y_id * surf_m.param.n + x_id)
        return grid_id

    def calc_xy_id_from_pos(surf_m, pos: float, lower_left: float, max_id: int):
        """
        calc_xy_id_from_pos: calculate x- or y-axis indices for given positional information

        :param pos: x- or y-axis position
        :param lower_left: lower left information
        :param max_id: max length (width or height)
        """
        id = int(np.floor((pos - lower_left) / surf_m.param.res))
        assert 0 <= id <= max_id, 'given position is out of the map!'
        return id

    def set_value_from_xy_pos(surf_m, x_pos: float, y_pos: float, val: float):
        """
        set_value_from_xy_pos: substitute given arbitrary values into data structure at specified x- and y-axis position

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        :param val: arbitrary spatial information
        """
        x_id, y_id = surf_m.get_xy_id_from_xy_pos(x_pos, y_pos)

        if (not x_id) or (not y_id):
            return False
        flag = surf_m.set_value_from_xy_id(x_id, y_id, val)
        return flag

    def set_value_from_xy_id(surf_m, x_id: int, y_id: int, val: float, field_name: str = "height", is_increment: bool = True):
        """
        set_value_from_xy_id: substitute given arbitrary values into data structure at specified x- and y-axis indices

        :param x_id: x index
        :param y_id: y index
        :param val: arbitrary spatial information
        :param field_name: structure's name
        :param is_increment: increment data if True. Otherwise, simply update value information.
        """
        if (x_id is None) or (y_id is None):
            return False, False
        grid_id = int(y_id * surf_m.param.n + x_id)

        if 0 <= grid_id < surf_m.param.num_grid:
            data = getattr(surf_m.data, field_name)
            if is_increment:
                data[grid_id] += val
            else:
                data[grid_id] = val
                setattr(surf_m.data, field_name, data)
            return True
        else:
            return False

    def extend_data(surf_m, data: np.array, field_name: str):
        """
        extend_data: extend surf_m.data in case additional terrain features are necessary

        :param data: appended data array
        :param field_name: structure's name
        """
        setattr(surf_m.data, field_name, data)

    # following functions are used for visualization objective
    def print_grid_map_info(surf_m):
        """
        print_grid_map_info: show grid map information

        """
        print("range: ", surf_m.param.n * surf_m.param.res, " [m]")
        print("resolution: ", surf_m.param.res, " [m]")
        print("# of data: ", surf_m.param.num_grid)

    def plot_maps(surf_m, figsize: tuple = (10, 4), field_name: str = "height"):
        """
        plot_maps: plot 2D and 2.5D figures with given size

        :param figsize: size of figure
        """
        sns.set()
        sns.set_style('whitegrid')
        fig = plt.figure(figsize=figsize)
        _, ax_3d = surf_m.plot_3d_map(fig=fig, rc=121, field_name=field_name)
        _, ax_2d = surf_m.plot_2d_map(fig=fig, rc=122, field_name=field_name)
        plt.tight_layout()

    def plot_2d_map(surf_m, grid_data: np.ndarray = None, fig: plt.figure = None, rc: int = 111, field_name: str = "height", 
                    cmap: str = "magma", label: str = "height m"):
        """
        plot_2d_map: plot 2D grid map

        :param grid_data: data to visualize
        :param fig: figure
        :param rc: position specification as rows and columns
        :param field_name: name of fields
        :param cmap: color map spec
        :param label: label of color map
        """
        xx, yy = np.meshgrid(np.arange(0.0, surf_m.param.n * surf_m.param.res, surf_m.param.res),
                             np.arange(0.0, surf_m.param.n * surf_m.param.res, surf_m.param.res))

        if grid_data is None:
            data = getattr(surf_m.data, field_name)
            if field_name == "height":
                grid_data = np.reshape(data, (surf_m.param.n, surf_m.param.n))
            elif field_name == "color":
                grid_data = data
            data = getattr(surf_m.data, field_name)
        else:
            data = np.reshape(grid_data, -1)
        if not fig:
            fig = plt.figure()

        ax = fig.add_subplot(rc)
        if field_name == "height":
            hmap = ax.pcolormesh(xx + surf_m.param.res / 2.0, yy + surf_m.param.res / 2.0, grid_data,
                                cmap=cmap, vmin=min(data), vmax=max(data))
            ax.set_xlim(xx.min(), xx.max() + surf_m.param.res)
            ax.set_ylim(yy.min(), yy.max() + surf_m.param.res)
            plt.colorbar(hmap, ax=ax, label=label, orientation='vertical')
        elif field_name == "color":
            hmap = ax.imshow(grid_data, origin='lower')
            ax.grid(False)
        ax.set_xlabel("x-axis m")
        ax.set_ylabel("y-axis m")
        ax.set_aspect("equal")

        return hmap, ax

    def plot_3d_map(surf_m, grid_data: np.ndarray = None, fig: plt.figure = None, rc: int = 111, field_name: str = "height"):
        """
        plot_3d_map: plot 2.5D grid map

        :param grid_data: data to visualize
        :param fig: figure
        :param rc: position specification as rows and columns
        """
        xx, yy = np.meshgrid(np.arange(0.0, surf_m.param.n * surf_m.param.res, surf_m.param.res),
                             np.arange(0.0, surf_m.param.n * surf_m.param.res, surf_m.param.res))

        if grid_data is None:
            grid_data = np.reshape(surf_m.data.height, (surf_m.param.n, surf_m.param.n))
            data = surf_m.data.height
        else:
            data = np.reshape(grid_data, -1)
        if not fig:
            fig = plt.figure()

        ax = fig.add_subplot(rc, projection="3d")
        if field_name == "height":
            hmap = ax.plot_surface(xx + surf_m.param.res / 2.0, yy + surf_m.param.res / 2.0, grid_data, rstride=1, cstride=1,
                                cmap="magma", vmin=min(data), vmax=max(data), linewidth=0, antialiased=False)
        elif field_name == "color":
            hmap = ax.plot_surface(xx + surf_m.param.res / 2.0, yy + surf_m.param.res / 2.0, grid_data, rstride=1, cstride=1,
                                facecolors=surf_m.data.color, linewidth=0, antialiased=False)
        ax.set_xlabel("x-axis m")
        ax.set_ylabel("y-axis m")
        ax.set_zticks(np.arange(xx.min(), xx.max(), 10))
        ax.view_init(elev=30, azim=45)
        ax.set_box_aspect((1, 1, 0.35))
        ax.set_xlim(xx.min(), xx.max() + surf_m.param.res)
        ax.set_ylim(yy.min(), yy.max() + surf_m.param.res)
        ax.set_zlim(min(data), xx.max() / 25)

        return hmap, ax


    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#For defining fractal and crater shapes


class Surface_Generation(grid_map):

    def __init__(surf_m, param, seed: int = 0):
        super().__init__(param, seed)
        surf_m.crater_prop = surf_m.param.crater_prop

    def set_terrain(surf_m):

#set_terrain: set planetary terrain environment based on fractal method w/o crater

 
        if surf_m.param.is_crater:
            if surf_m.crater_prop.distribution == "single":
                c_xy_ = np.array([int(surf_m.c_x), int(surf_m.c_y)]).reshape(1, 2)
                D = surf_m.crater_prop.min_D
                surf_m.set_crater(c_xy=c_xy_, D=D)
            elif surf_m.crater_prop.distribution == "random":
                i = 0
                if surf_m.crater_prop.con_D is not None:
                    num_crater = surf_m.calculate_num_crater(D=surf_m.crater_prop.con_D)
                else:
                    num_crater = surf_m.calculate_num_crater(D=surf_m.crater_prop.max_D)
                while i < num_crater:
                    c_xy_ = surf_m.rng.integers(surf_m.lower_left_x, (surf_m.param.n - 1) * surf_m.param.res, 2).reshape(1, 2)
                    if surf_m.crater_prop.con_D is not None:
                        D = surf_m.crater_prop.con_D
                    else:
                        D = surf_m.rng.integers(surf_m.crater_prop.min_D, surf_m.crater_prop.max_D)
                    if i == 0:
                        surf_m.set_crater(c_xy=c_xy_, D=D)
                        # init array for checking circle hit
                        c_arr = c_xy_
                        d_arr = np.array([D])
                        i += 1
                    else:
                        is_hit = surf_m.check_circle_hit(c_arr, d_arr, c_xy_, D)
                        if not is_hit:
                            surf_m.set_crater(c_xy=c_xy_, D=D)
                            c_arr = np.append(c_arr, c_xy_, axis=0)
                            d_arr = np.append(d_arr, D)
                            i += 1
        if surf_m.param.is_fractal:
            surf_m.set_fractal_surf()
        surf_m.data.height = surf_m.set_offset(surf_m.data.height)

    def check_circle_hit(surf_m, c_arr: np.ndarray, d_arr: np.ndarray, c_t: np.ndarray, d_t: np.ndarray):
        """
        check_circle_hit: check whether given craters are overlapped or not
        
        :param c_arr: center positions of generated craters so far
        :param d_arr: diameter information of generated creaters so far
        :param c_t: center position of newly generated crater
        :param d_t: diameter information of newly generated crater
        """
        
        for c, d, in zip(c_arr, d_arr):
            dist_c = np.sqrt((c[0] - c_t[0, 0])**2 + (c[1] - c_t[0, 1])**2)
            sum_d = (d + d_t)
            if dist_c < sum_d:
                return True
        return False
        
    def set_fractal_surf(surf_m):
        
#Set fractal surface into data structure 

       
        z = surf_m.generate_fractal_surf()
        # set offset
        z = surf_m.set_offset(np.ravel(z))
        surf_m.data.height += z

    def set_crater(surf_m, c_xy: np.ndarray, D: float):
        """
        set_crater: set arbitrary crater generated with given parameters into 2D map environment

        :param c_xy: center of crater position in x- and y-axis [m]
        :param D: crater inner-rim range
        """
        z = surf_m.generate_crater(D)
        x_c_id, y_c_id = surf_m.get_xy_id_from_xy_pos(c_xy[0, 0], c_xy[0, 1])
        x_len = int(z.shape[0] / 2)
        y_len = int(z.shape[1] / 2)
        for y_id_ in range(y_c_id - y_len, y_c_id + y_len):
            for x_id_ in range(x_c_id - x_len, x_c_id + x_len):
                if surf_m.lower_left_x <= x_id_ < surf_m.param.n and surf_m.lower_left_y <= y_id_ < surf_m.param.n:
                    surf_m.set_value_from_xy_id(x_id_, y_id_, z[x_id_ - (x_c_id - x_len), y_id_ - (y_c_id - y_len)])

    def set_offset(surf_m, z: np.ndarray):
        """
        set_offset: adjust z-axis value starting from zero

        :param z: z-axis information (typically for distance information, such as terrain height)
        """
        z_ = z - min(z)
        z = (z_ + abs(z_)) / 2
        z_ = z - max(z)
        z_ = (z_ - abs(z_)) / 2
        z = z_ - min(z_)
        return z
            
    def generate_fractal_surf(surf_m):
        """
Generates random height information based on fractional Brownian motion (fBm).

        """
        z = np.zeros((surf_m.param.n, surf_m.param.n), complex)
        for y_id_ in range(int(surf_m.param.n / 2) + 1):
            for x_id_ in range(int(surf_m.param.n / 2) + 1):
                phase = 2 * np.pi * surf_m.rng.random()
                if x_id_ != 0 or y_id_ != 0:
                    rad = 1 / (x_id_**2 + y_id_**2)**((surf_m.param.re + 1) / 2)
                else:
                    rad = 0.0
                z[y_id_, x_id_] = rad * np.exp(1j * phase)
                if x_id_ == 0:
                    x_id_0 = 0
                else:
                    x_id_0 = surf_m.param.n - x_id_
                if y_id_ == 0:
                    y_id_0 = 0
                else:
                    y_id_0 = surf_m.param.n - y_id_
                z[y_id_0, x_id_0] = np.conj(z[y_id_, x_id_])

        z[int(surf_m.param.n / 2), 0] = np.real(z[int(surf_m.param.n / 2), 0])
        z[0, int(surf_m.param.n / 2)] = np.real(z[0, int(surf_m.param.n / 2)])
        z[int(surf_m.param.n / 2), int(surf_m.param.n / 2)] = np.real(z[int(surf_m.param.n / 2), int(surf_m.param.n / 2)])

        for y_id_ in range(1, int(surf_m.param.n / 2)):
            for x_id_ in range(1, int(surf_m.param.n / 2)):
                phase = 2 * np.pi * surf_m.rng.random()
                rad = 1 / (x_id_ ** 2 + y_id_ ** 2) ** ((surf_m.param.re + 1) / 2)
                z[y_id_, surf_m.param.n - x_id_] = rad * np.exp(1j * phase)
                z[surf_m.param.n - y_id_, x_id_] = np.conj(z[y_id_, surf_m.param.n - x_id_])

        z = z * abs(surf_m.param.sigma) * (surf_m.param.n * surf_m.param.res * 1e+3)**(surf_m.param.re + 1 + .5)
        z = np.real(fftpack.ifft2(z)) / (surf_m.param.res * 1e+3)**2
        z = z * 1e-3
        return z

    def generate_crater(surf_m, D: np.ndarray):
        """
        generate_crater: generate crater height information

        :param D: crater inner-rim range
        """
        xx, yy = np.meshgrid(np.arange(0.0, int(2.0 * D * 2.5), surf_m.param.res),
                             np.arange(0.0, int(2.0 * D * 2.5), surf_m.param.res))
        xx, yy = xx / D, yy / D
        c_x = c_y = 2.5
        rr = np.sqrt((xx - c_x)**2 + (yy - c_y)**2)
        zz = np.full((rr.shape[0], rr.shape[1]), np.nan)
        rmax = - np.inf
        hout = np.nan
        for y_id_, x_id_ in np.ndindex(rr.shape):
            r = rr[y_id_, x_id_]
            if surf_m.crater_prop.geometry == "normal":
                h = surf_m.normal_model(r, D)
            elif surf_m.crater_prop.geometry == "mound":
                h = surf_m.central_mound_crater_model(r, D)
            elif surf_m.crater_prop.geometry == "flat":
                h = surf_m.flat_bottomed_crater_model(r, D)
            elif surf_m.crater_prop.geometry == "concentric":
                h = surf_m.concentric_crater_model(r, D)
            zz[y_id_, x_id_] = h
            if rmax < r:
                rmax = r
                hout = h
        zz[zz == np.nan] = hout
        zz -= hout
        zz *= D
        return zz

    def generate_RGB_info(surf_m, t, t_upper):
        """
        generate_RGB_info: generate RGB information including terrain appearance with shading condition

        """
        surf_m.data.color = np.full((surf_m.param.n, surf_m.param.n, 3), 0.8)
        surf_m.create_shading(t, t_upper)

    def create_shading(surf_m, t: float, t_upper: float = 1, ambient: float = 0.1):
        """
        create_shading: create shading effect using color and height information

        """
        # get relevant data from the data structure
        height = np.reshape(surf_m.data.height, (surf_m.param.n, surf_m.param.n))
        color = surf_m.data.color.transpose(2, 0, 1).astype(np.float32)
        # calculate normal vector
        dx = height[:, :-1] - height[:, 1:]
        dy = height[:-1, :] - height[1:, :]
        norm = np.zeros((1, 3, surf_m.param.n, surf_m.param.n))
        norm[:, 0, :, :-1] += dx
        norm[:, 0, :, 1:] += dx
        norm[:, 0, :, 1:-1] /= 2
        norm[:, 1, :-1, :] += dy
        norm[:, 1, 1:, :] += dy
        norm[:, 1, 1:-1, :] /= 2
        norm[:, 2] = 1
        norm /= (norm * norm).sum(axis=1, keepdims=True)**(0.5)
        # generate light source vector
        theta = surf_m.rng.random() * (2 * np.pi)
        z = surf_m.rng.random() * (t_upper - t) + t
        r = (1 - z * z)**(0.5)
        #Coordinate
        l = np.zeros((1, 3))
        l[:, 0] = r * np.cos(theta)
        l[:, 1] = r * np.sin(theta)
        l[:, 2] = z
        # cast shading to color image
        shade = (l[:, :, None, None] * norm).sum(axis=1, keepdims=True)
        shade = np.clip(shade, 0, None) * np.expand_dims(color, 0)
        shade += np.expand_dims(color, 0) * ambient
        surf_m.data.color = np.squeeze(shade).transpose(1, 2, 0)

    def normal_model(surf_m, r, D):
        """
        normal_model: normal crater model

        :param r: input variable
        :param D: diameter
        """
        d0 = 0.114 * D**(-0.002)
        hr = 0.02513 * D**(-0.0757)
        alpha = -3.1906
        
        a = -2.8567
        b = 5.8270
        C = d0 * ((np.exp(a) + 1) / (np.exp(b) - 1))
        if 0 <= r <= 1.0:
            h = C * (np.exp(b * r) - np.exp(b)) / (1 + np.exp(a + b * r))
        else:
            h = hr * (r**alpha - 1)
        return h

    def central_mound_crater_model(surf_m, r, D):
        """
        central_mound_crater_model: central mound crater model

        :param r: input variable
        :param D: diameter
        """
        d0 = 0.114 * D**(-0.002)
        hr = 0.02513 * D**(-0.0757)
        alpha = -3.1906
        
        rm = 0.293 * D**(-0.086)
        rb = 0.793 * D**(-0.242)
        hm = 0.23 * 10**(-3) * D**(0.64)
        a = -2.6921
        b = 6.1678
        C = d0 * ((np.exp(a) + 1) / (np.exp(b) - 1))
        if 0 <= r <= rm:
            h = (1 - (r / rm)) * hm - d0
        elif rm < r <= rb:
            h = -d0
        elif rb < r <= 1:
            r0 = (r - rb) / (1 - rb)
            h = C * ((np.exp(b * r0) - np.exp(b)) / (1 + np.exp(a + b * r0)))
        else:
            h = hr * (r**alpha - 1)
        return h

    def flat_bottomed_crater_model(surf_m, r, D):
        """
        flat_bottomed_crater_model: flat-bottomed crater model

        :param r: input variable
        :param D: diameter
        """
        d0 = 0.114 * D**(-0.002)
        hr = 0.02513 * D**(-0.0757)
        alpha = -3.1906
        
        rb = 0.091 * D**(0.208)
        a = -2.6003
        b = 5.8783
        C = d0 * ((np.exp(a) + 1) / (np.exp(b) - 1))
        if 0 <= r <= rb:
            h = -d0
        elif rb < r <= 1:
            r0 = (r - rb) / (1 - rb)
            h = C * ((np.exp(b * r0) - np.exp(b)) / (1 + np.exp(a + b * r0)))
        else:
            h = hr * (r**alpha - 1)
        return h

    def concentric_crater_model(surf_m, r, D):
        """
        concentric_crater_model: concentric crater model

        :param r: input variable
        :param D: diameter
        """
        d0 = 0.114 * D**(-0.002)
        hr = 0.02513 * D**(-0.0757)
        alpha = -3.1906
        
        C1 = 0.1920
        C2 = 0.0100
        C3 = 0.0155 * D**(0.343)
        ri = 0.383 * D**(0.053)
        ro = 0.421 * D**(0.102)
        a = -1.6536
        b = 4.7626
        h1 = np.nan
        h2 = np.nan
        if 0 <= r <= ri:
            h = C1 * r**2 + C2 * r - d0
        elif ri < r <= ro:
            h = C3 * (r - ri) + h1
        elif ro < r <= 1:
            C = - h2 * ((np.exp(a) + 1) / (np.exp(b) - 1))
            r0 = (r - ro) / (1 - ro)
            h = C * ((np.exp(b * r0) - np.exp(b)) / (1 + np.exp(a + b * r0)))
        else:
            h = hr * (r**alpha - 1)
        return h

    def calculate_num_crater(surf_m, D: float, a: float = 1.54, b: float = - 2):
        """
        calculate_num_crater: calculate number of craters based on analytical model

        :param D: diameter
        :param a: coefficient of the function
        :param b: coefficient of the function
        """
        density = a * D**b
        area_t = (surf_m.param.n * surf_m.param.res)**2
        area_c = (D / 2)**2 * np.pi
        num_crater = int((area_t) * density / area_c)
        return num_crater

    
    
    
    

