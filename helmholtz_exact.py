import numpy as np
from scipy.special import hankel1 as H1, jn as J, h1vp as dH1, jvp as dJ

import numpy.typing as npt

complex_array = npt.NDArray[np.complex128]
float_array = npt.NDArray[np.float64]


def Det(A_11: complex_array, A_21: complex_array,
        A_12: complex_array, A_22: complex_array) -> complex_array:
    """Computes the determinant of a two-by-two matrix given
    by its elements. Inputs are allowed to be arrays of the same shape"""
    return A_11 * A_22 - A_12*A_21


def DielectricPlaneWaveCoefficients(k: float, N: float, R: float, xy_c: float_array, U: complex, theta_inc: float, M: int) -> tuple[complex_array, complex_array]:
    """Computes the coefficientes of the Bessel expansion of the total field
    inside the scatterer and the scattered field outside for an incident
    plane wave.

    Inputs:
    - k: wavenumber of the background
    - N: index of refraction of the scatterer with respect to the background
    - c: center of the circular scatterer
    - R: radius of the circular scatterer
    - U: complex amplitude of the incident plane wave
    - theta_inc: angles that the propagating direction of the incident plane wave
    forms with the x-axis.
    - M: number of modes used in the expansion, i.e. n = -M, -M+1, ..., M-1, M
    Outputs:
    - A: coefficients of the scattered field outside the scatterer:
        u_s(r, theta) = sum_{n=-M}^M a_n H^1_n*(k*r)*exp(i*n*theta)
    - B: coefficients of the total field inside the scatterer:
        u(r, theta) = sum_{n=-M}^M b_n*J_n(k*r)*exp(i*n*theta)"""

    # vectorized version
    n = np.arange(-M, M+1, dtype=np.int64)
    dx = np.cos(theta_inc)
    dy = np.sin(theta_inc)
    W = Det(H1(n, k*R), dH1(n, k*R), -J(n, np.sqrt(N)*k*R), -np.sqrt(N)*dJ(n, np.sqrt(N)*k*R))
    A = -U*np.exp(1j*k*(dx*xy_c[0] + dy*xy_c[1]))*np.exp(-1j*n*theta_inc)*1j**n*Det(J(n, k*R), dJ(n, k*R), -J(n,np.sqrt(N)*k*R), -np.sqrt(N)*dJ(n,np.sqrt(N)*k*R))/W        
    B = -U*np.exp(1j*k*(dx*xy_c[0] + dy*xy_c[1]))*np.exp(-1j*n*theta_inc)*1j**n*Det(H1(n, k*R), dH1(n, k*R), J(n,k*R), dJ(n,k*R))/W
    
    return (A, B) 


def DielectricHankelCoefficients(k: float, N: float, R: float, xy_c: float_array, U: complex, r_E: float_array, M: int) -> tuple[complex_array, complex_array]:
    """Computes the coefficientes of the Bessel expansion of the total field
    inside the scatterer and the scattered field outside for an incident
    Hankel wave.

    Inputs:
    - k: wavenumber of the background
    - N: index of refraction of the scatterer with respect to the background
    - c: center of the circular scatterer
    - R: radius of the circular scatterer
    - U: complex amplitude of the incident Hankel wave
    - r_E: location of the emitter.
    - M: number of modes used in the expansion, i.e. n = -M, -M+1, ..., M-1, M
    Outputs:
    - A: coefficients of the scattered field outside the scatterer:
        u_s(r, theta) = sum_{n=-M}^M a_n H^1_n*(k*r)*exp(i*n*theta)
    - B: coefficients of the total field inside the scatterer:
        u(r, theta) = sum_{n=-M}^M b_n*J_n(k*r)*exp(i*n*theta)"""
    r_cE = np.sqrt((r_E[0] - xy_c[0])**2 + (r_E[1] - xy_c[1])**2)
    theta_E = np.atan2(r_E[1], r_E[0])
    # vectorized version
    n = np.arange(-M, M+1, dtype=np.int64)
    W = Det(H1(n, k*R), dH1(n, k*R), -J(n, np.sqrt(N)*k*R), -np.sqrt(N)*dJ(n, np.sqrt(N)*k*R))
    A = -U*H1(n, k*r_cE)*np.exp(-1j*n*theta_E)*Det(J(n, k*R), dJ(n, k*R), -J(n,np.sqrt(N)*k*R), -np.sqrt(N)*dJ(n,np.sqrt(N)*k*R))/W        
    B = -U*H1(n, k*r_cE)*np.exp(-1j*n*theta_E)*Det(H1(n, k*R), dH1(n, k*R), J(n,k*R), dJ(n,k*R))/W
    
    return (A, B) 






def ConductingPlaneWaveCoefficients(k: float, R: float, xy_c: float_array, U: complex, theta_inc: float, M: int) -> tuple[complex_array, complex_array]:
    """Computes the coefficientes of the Bessel expansion of the scattered field outside
    a conducting scatterer for an incident plane wave.

    Inputs:
    - k: wavenumber of the background
    - c: center of the circular scatterer
    - R: radius of the circular scatterer
    - U: complex amplitude of the incident plane wave
    - theta_inc: angles that the propagating direction of the incident plane wave
    forms with the x-axis.
    - M: number of modes used in the expansion, i.e. n = -M, -M+1, ..., M-1, M
    Outputs:
    - A: coefficients of the scattered field outside the scatterer:
        u_s(r, theta) = sum_{n=-M}^M a_n H^1_n*(k*r)*exp(i*n*theta)"""

    # vectorized version
    n = np.arange(-M, M+1, dtype=np.int64)
    dx = np.cos(theta_inc)
    dy = np.sin(theta_inc)
    A = -U*np.exp(1j*k*(dx*xy_c[0] + dy*xy_c[1]))*np.exp(-1j*n*theta_inc)*1j**n*J(n, k*R)/H1(n, k*R)
    
    return A







def ConductingHankelCoefficients(k: float, R: float, xy_c: float_array, U: complex, r_E: float_array, M: int) -> tuple[complex_array, complex_array]:
    """Computes the coefficientes of the Bessel expansion of the scattered field outside
    a conducting scatterer for an incident Hankel wave.

    Inputs:
    - k: wavenumber of the background
    - c: center of the circular scatterer
    - R: radius of the circular scatterer
    - U: complex amplitude of the incident plane wave
    - r_E: location of the emitter.
    - M: number of modes used in the expansion, i.e. n = -M, -M+1, ..., M-1, M
    Outputs:
    - A: coefficients of the scattered field outside the scatterer:
        u_s(r, theta) = sum_{n=-M}^M a_n H^1_n*(k*r)*exp(i*n*theta)"""

    r_cE = np.sqrt((r_E[0] - xy_c[0])**2 + (r_E[1] - xy_c[1])**2)
    theta_E = np.atan2(r_E[1], r_E[0])

    # vectorized version
    n = np.arange(-M, M+1, dtype=np.int64)
    A = -H1(n, k*r_cE)/H1(n, k*R)*J(n, k*R)*np.exp(-1j*n*theta_E)
    
    return A







def PlaneWave(X: float_array, Y: float_array, k: float, d: float_array, U: complex = 1 + 0j) -> complex_array: 
    """
    Evaluates a plane wave field in all the (x,y) points given
    """
    return U*np.exp(1j*k*(d[0]*X + d[1]*Y))

# def Fundamental(X: float_array,
#                 Y: float_array,
#                 k: float,
#                 x_s: float,
#                 y_s: float,
#                 U: complex = 1.+0.j) -> complex_array:
#     """Evaluates a point source wave"""
#     return U*1j/4  * H1(0, k*np.hypot(X-x_s, Y-y_s))

def HankelWave(X: float_array,
                Y: float_array,
                k: float,
                x_s: float,
                y_s: float,
                U: complex = 1.+0.j) -> complex_array:
    """Evaluates a point source wave"""
    return U * H1(0, k*np.hypot(X-x_s, Y-y_s))


def U_tot_from_coefficients(X: float_array, Y: float_array, k: float, N: float,
                      c: float_array, R: float, U: complex,
                      U_inc: complex_array, A: complex_array, B: complex_array) -> complex_array:
    M = (len(A)-1)//2
    n = np.arange(-M, M+1, dtype=np.int64)
    r = np.hypot(X-c[0], Y-c[1])
    n = np.expand_dims(n, axis=np.arange(X.ndim).tolist())
    r = np.expand_dims(r, axis=-1)
    theta = np.arctan2(Y-c[1], X-c[0])
    theta = np.expand_dims(theta, axis=-1)
    U_in  = np.dot(J(n,np.sqrt(N)*k*r)*np.exp(1j*n*theta), B)
    U_out = U_inc + np.dot(H1(n, k*r)*np.exp(1j*n*theta), A)
    r = np.squeeze(r)
    U_tot = np.where(r > R, U_out, U_in)
    return U_tot

def U_tot_from_coefficients_conducting(X: float_array, Y: float_array, k: float,
                      c: float_array, R: float, U: complex,
                      U_inc: complex_array, A: complex_array) -> complex_array:
    M = (len(A)-1)//2
    n = np.arange(-M, M+1, dtype=np.int64)
    r = np.hypot(X-c[0], Y-c[1])
    n = np.expand_dims(n, axis=np.arange(X.ndim).tolist())
    r = np.expand_dims(r, axis=-1)
    theta = np.arctan2(Y-c[1], X-c[0])
    theta = np.expand_dims(theta, axis=-1)
    U_out = U_inc + np.dot(H1(n, k*r)*np.exp(1j*n*theta), A)
    r = np.squeeze(r)
    U_tot = np.where(r > R, U_out, np.full_like(X,fill_value=np.nan))
    return U_tot




def mear_field_plane_wave(xy_E: float_array, xy_R: float_array, k: float, R: float, c: float_array, M: int) -> complex_array:
    """I don't like this implementation, as if you emmit from a given point
    your incident field should not be a plane wave"""
    pass


def far_field_from_plane_wave(theta_E: float_array, theta_R: float_array, U: complex, k: float, R: float, c: float_array, N: float, M: int) -> complex_array:
    N_E = len(theta_E)
    N_R = len(theta_R)
    FF = np.zeros((N_R, N_E), dtype=np.complex128)
    n = np.arange(-M, M+1)
    n = np.expand_dims(n, 0)
    theta = np.expand_dims(theta_R, -1)
    x_hat = np.cos(theta)
    y_hat = np.sin(theta)
    for j, theta_inc in enumerate(theta_E):
        A, _ = DielectricPlaneWaveCoefficients(k, N, R, c, U, theta_inc, M)
        FF[:, j] = np.sqrt(2/np.pi/k)*np.dot(np.exp(1j*(n*(theta - np.pi/2) - np.pi/4 - k*(c[0]*x_hat + c[1]*y_hat))), A)

    return FF





if __name__ == "__main__":
    pass
