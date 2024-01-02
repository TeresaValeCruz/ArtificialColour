import numpy as np

"""he transfer functions between sRGB and hex"""


def sRGB2HEX(array):
    return '#{0:02x}{1:02x}{2:02x}'.format(*array)


def HEX2sRGB(string):
    return np.array([int(string[k:k + 2], 16) for k in [1, 3, 5]])


"""the transfer functions between sRGB and linear RGB, as defined by the ICC"""


def standard2linear(color):
    """from standard 24-bit RGB (sRGB) to linear RGB"""
    norm_color = color / 255
    if norm_color <= 0.04045:
        linear_color = norm_color / 12.92
    else:
        linear_color = ((norm_color + 0.055) / 1.055) ** 2.4
    return linear_color


def linear2standard(linear_color):
    """from linear RGB to standard 24-bit RGB (sRGB)"""
    if linear_color <= 0.0031308:
        norm_color = linear_color * 12.92
    else:
        norm_color = ((linear_color ** (1 / 2.4)) * 1.055) - 0.055
    color = round(norm_color * 255)
    return color


"""functions that convert linear RGB into D65 XYZ and vice versa, as defined by the ICC"""

M = np.matrix([[12831 / 3959, -329 / 214, -1974 / 3959],
               [-851781 / 878810, 1648619 / 878810, 36519 / 878810],
               [705 / 12673, -2585 / 12673, 705 / 667]])

N = np.matrix([[506752 / 1228815, 87881 / 245763, 12673 / 70218],
               [87098 / 409605, 175762 / 245763, 12673 / 175545],
               [7918 / 409605, 87881 / 737289, 1001167 / 1053270]])


def XYZ2RGB(array):
    """from CIE XYZ to linear RGB"""
    t_array = np.atleast_2d(array).T
    rgb = M * t_array
    return np.asarray(rgb.T)[0]


def RGB2XYZ(array):
    """from linear RGB to CIE XYZ"""
    t_array = np.atleast_2d(array).T
    xyz = N * t_array
    return np.asarray(xyz.T)[0]


"""functions that convert D65 XYZ into CIE Luv and vice versa, as defined in Measuring Color by R.Hunt and M.Pointer"""
"""CIE D65 is the illuminant"""

white_chroma = np.array([0.31271, 0.32902, 0.35827])  # values for x, y and z; they add to 1

w = white_chroma
reference_white = np.array([w[0] / w[1], w[1] / w[1], w[2] / w[1]])


def u_line(array):
    """input XYZ array"""
    X, Y, Z = array[0], array[1], array[2]
    if X == 0:
        return 0
    else:
        return 4 * X / (X + 15 * Y + 3 * Z)


def v_line(array):
    """input XYZ array"""
    X, Y, Z = array[0], array[1], array[2]
    if Y == 0:
        return 0
    else:
        return 9 * Y / (X + 15 * Y + 3 * Z)


def XYZ2LUV(array):
    """from XYZ to Luv"""
    # input normalized XYZ array
    X, Y, Z = array[0], array[1], array[2]

    y_fraction = Y / reference_white[1]
    if y_fraction <= (6 / 29) ** 3:
        L = ((29 / 3) ** 3) * y_fraction
    else:
        L = 116 * (y_fraction ** (1 / 3)) - 16
    u = 13 * L * (u_line(array) - u_line(reference_white))
    v = 13 * L * (v_line(array) - v_line(reference_white))
    return np.array([L, u, v])


def LUV2XYZ(array):
    """from Luv to XYZ"""
    # input Luv array
    L, u, v = array[0], array[1], array[2]

    if L == 0:
        return np.array([0, 0, 0])
    if L <= 8:
        Y = reference_white[1] * L * (3 / 29) ** 3
    else:
        Y = reference_white[1] * ((L + 16) / 116) ** 3
    temp_U = u / (13 * L) + u_line(reference_white)
    temp_V = v / (13 * L) + v_line(reference_white)
    X = Y / (4 * temp_V) * 9 * temp_U
    Z = Y / (4 * temp_V) * (12 - 3 * temp_U - 20 * temp_V)
    return np.array([X, Y, Z])


"""from orthogonal Luv to Polar Luv and vice-versa"""
rad2deg = 180/np.pi
deg2rad = np.pi/180

def LUV2LCH(array):
    """input a Luv triple and outputs a LCh triple"""
    u_sq, v_sq = array[1]**2, array[2]**2
    C = np.sqrt(u_sq+v_sq)
    radian_H = np.arctan2(array[2], array[1])
    if radian_H < 0:
        radian_H += 2*np.pi
    return np.array([array[0], C, radian_H * rad2deg])

def LCH2LUV(array):
    """input a LCh triple and outputs a Luv triple"""
    radian_H = array[2]*deg2rad
    u = array[1]*np.cos(radian_H)
    v = array[1]*np.sin(radian_H)
    return np.array([array[0], u, v])


"""finally, functions to transform sRGB to CIE Luv and vice-versa"""


def sRGB2LUV(array):
    """input standard 24-bit RGB (sRGB)"""
    linearize = np.vectorize(standard2linear)
    linearRGB = linearize(array)
    XYZ = RGB2XYZ(linearRGB)
    LUV = XYZ2LUV(XYZ)
    return LUV


def LUV2sRGB(array):
    """input CIE Luv"""
    XYZ = LUV2XYZ(array)
    linearRGB = XYZ2RGB(XYZ)
    standardize = np.vectorize(linear2standard)
    sRGB = standardize(linearRGB)
    return sRGB


def HEX2LUV(string):
    """input standard 24-bit RGB (sRGB)"""
    sRGB = HEX2sRGB(string)
    linearize = np.vectorize(standard2linear)
    linearRGB = linearize(sRGB)
    XYZ = RGB2XYZ(linearRGB)
    LUV = XYZ2LUV(XYZ)
    return LUV


def LUV2HEX(array):
    """input CIE Luv"""
    XYZ = LUV2XYZ(array)
    linearRGB = XYZ2RGB(XYZ)
    standardize = np.vectorize(linear2standard)
    sRGB = standardize(linearRGB)
    return sRGB2HEX(sRGB)
