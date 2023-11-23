"""Shock simulation constants"""
import numpy as np

PATH_TO_RESULT = "./result"
PATH_TO_PLOT = "./plots"
STEP_INIT = 0

C = 0.5
QE = -0.0055901699
QI = -QE
ME = 1.0
MI = 100.0

LSI = 200.0
RES = 4.0
RES_PHASE = 5.0
V0 = -0.1

ION_GYROTIME = 48001.

THETA = 60.0
PHI = 90.0

N0 = 20.0

b0 = 0.3726779
e0 = -V0*b0
B0 = b0*np.array([np.cos(THETA*np.pi/180.0),
                  np.sin(THETA*np.pi/180.0)*np.cos(PHI*np.pi/180.0),
                  np.sin(THETA*np.pi/180.0)*np.sin(PHI*np.pi/180.0)
                  ])
E0 = -V0*np.array([0.0, -B0[2], B0[1]])

DIGITS = 6
WPE = 0.025

MMX = int(12*96 // RES)
MMY = int(12*96 // RES)

