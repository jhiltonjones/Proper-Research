import numpy as np 
import matplotlib.pyplot as plt



def magnetic_moment(B_r, mu0, r, length):
    term1 = B_r / mu0
    term2 = np.pi * (r**2) * length
    return term1 * term2
def magnetic_field(mu_0, mu, r, mu_hat):
    r_mag = np.linalg.norm(r)
    r_hat = r/np.linalg.norm(r)
    term1 = (mu_0 * mu)/(4* np.pi * r_mag**3)
    term2 = (3*np.outer(r_hat,r_hat))-np.eye(3)
    return ( term1 * term2) @ mu_hat
def magnetic_field_alpha_phi(phi, alpha, R, m0, mu0):
   r_vec = np.array([R* np.cos(phi), R*np.sin(phi), 0])
   R_norm = np.linalg.norm(r_vec)
   r_hat = r_vec/R_norm

   m0_vec = m0 * r_hat
   c,s = np.cos(alpha), np.sin(alpha)
   R_z = np.array([
       [c,-s,0],
       [s,c,0],
       [0,0,1]
   ])
   m_vec = R_z@m0_vec
   m_dot_r = np.dot(m_vec, r_hat)
   B_vec = (mu0/(4*np.pi*R_norm**3))*(3*m_dot_r*r_hat-m_vec)
   return B_vec

def B_and_beta_from_phi_and_alpha(phi, alpha, R, m0, mu0):
    B_vec = magnetic_field_alpha_phi(phi, alpha, R, m0, mu0)
    Bx, By, Bz = B_vec
    B_mag = np.linalg.norm(B_vec)
    beta = np.arctan2(By, Bx)
    return B_mag, beta

