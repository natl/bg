################################################################################
#
# Program:      Initial state creation for trap simulation
#
# Author:       Nathanael Lampe, School of Physics, Monash University
#
# Created:      December 4, 2012
#
# Changelog:    December 4, 2012: Created from twodtrap.py
#           
#               
#
# Purpose:      Contain functions that create initial states in twod.py
#               
################################################################################

#Imports
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import det
from scipy.special import j0, j1, jn_zeros
from numpy.random import randn

def gravground( x          ,
                y          ,
                g          ,
                G          ,
                *args      ,
                **kwargs):
  '''
  def gravground( x       ,
                  y       ,
                  g       ,
                  G       ,
                  *args   ,
                  **kwargs):
  Create the Thomas-Fermi ground state for a gravitational system
  '''
  X,Y = np.meshgrid(x,y)
  R = np.sqrt( X ** 2. + Y ** 2. )
  bj0z1   = jn_zeros( 0, 1 ) #First zero of zeroth order besselj
  scaling = np.sqrt( 2 * np.pi * G / g  )
  gr0     = bj0z1 / scaling
  Rprime  = R * scaling
  
  gtfsol = j0( Rprime ) * np.array( [ map( int,ii ) for ii in map( 
                                      lambda rad: rad <= gr0, R ) ] )

      
  gtfsol *= scaling ** 2. / ( 2 * np.pi * j1( bj0z1 ) * bj0z1 ) 
  
  return gtfsol
  
def evenvortex( rad, num ):
  '''
  return a list of 1x3 lists of vortex positions
  The third column defaults to 1, specifying vortex 
  angular momentum and orientation.
  There is no way to change this currently
  '''
  ba = 2*np.pi / num
  l = []
  for j in range(0,num):
    l.append( [rad * np.cos( j * ba ), rad * np.sin( j * ba ), 1. ])
  return l

################################################################################

def lastpsi(x,psi):
  return psi

################################################################################

def twoblobs(x        ,
             y        ,
             *args    ,
             **kwargs ):
  X,Y = np.meshgrid(x,y)
  xo1 = -2.
  xo2 = +2.
  sig1 = 1.
  pi = np.pi
  sig2 = 1.
  psi1 = 1.0/np.sqrt(2*pi*sig1**2) * np.exp(-((Y-xo1)**2 + X**2.) / (2*sig1**2))\
         +.0001*randn(np.size( X )).reshape( [ np.size(X,0), np.size(X,1) ] )\
         * np.exp(-((Y-xo1)**2 + X**2.) / (2*sig1**2))
  psi2 = 1.0/np.sqrt(2*pi*sig2**2) * np.exp(-((Y-xo2)**2 + X**2.) / (2*sig2**2))\
         +.0001*randn(np.size( X )).reshape( [ np.size(X,0), np.size(X,1) ] )\
         * np.exp(-((Y-xo2)**2 + X**2.) / (2*sig2**2))
  psi = np.zeros( [ len(x), len(y) ], dtype=complex )
  psi += psi1 + psi2
  norm = sum(sum(psi*psi.conjugate())) * abs(x[1]-x[0]) * abs(y[1]-y[0])
  psi = psi / np.sqrt(norm)
  return psi

################################################################################
  
def threeblobs(x        ,
               y        ,
               *args    ,
               **kwargs ):
  X,Y = np.meshgrid(x,y)
  xo1 = 0.
  yo1 = 4.
  xo2 = 2. * np.sqrt(3.)
  yo2 = -2.
  xo3 = -2. * np.sqrt(3.)
  yo3 = -2.
  sig1 = 1.
  sig2 = 1.
  sig3 = 1.
  pi = np.pi
  psi1 = 1.0 / np.sqrt(2*pi*sig1**2) * np.exp(
                           -((X-xo1)**2 + (Y-yo1)**2.) / (2*sig1**2)) 
  psi2 = 1.0 / np.sqrt(2*pi*sig2**2) * np.exp(
                           -((X-xo2)**2 + (Y-yo2)**2.) / (2*sig2**2)) 
  psi3 = 1.0 / np.sqrt(2*pi*sig3**2) * np.exp(
                           -((X-xo3)**2 + (Y-yo3)**2.) / (2*sig3**2)) 
  psi = psi1 + psi2 + psi3
  norm = sum(sum(psi*psi.conjugate())) * abs(x[1]-x[0]) * abs(y[1]-y[0])
  psi = psi / np.sqrt(norm)
  return psi

################################################################################
  
def gauss( x              ,
           y              ,
           means = [0.,0.],
           sig   = [1.,1.],
           corr  = 0.     ,
           *args          ,
           **kwargs       ):
  '''
  def gauss( x              ,
             y              ,
             means = [0.,0.],
             sig   = [1.,1.],
             corr  = 0.     ):
  Define an initial wavefunction (Gaussian)
  '''
  if abs( corr ) >= 1: print '<twod.gauss> Error: ensure |corr| < 1'
  X, Y = np.meshgrid( x, y )
  return ( 1. / ( 2. * np.pi * sig[0] * sig[1] * np.sqrt( 1 - corr **2 ) ) *
        np.exp( -1. / ( 2. * ( 1 - corr **2 ) ) *
          ( ( X - means[0] ) ** 2. / sig[0] +
            ( Y - means[1] ) ** 2. / sig[1] - 
            2 * corr * ( X - means[0] ) * ( Y - means[1] ) / ( sig[0] * sig[1] )
          )   )
         )

################################################################################
  
def vortexgauss( x                 ,  #xvals
                 y                 ,  #yvals
                 vort  = [0, 0, 0] ,  #X by 2. array of vortex locations
                 means = [0.,0.]   ,  #[x,y] centre of gaussian
                 sig   = [1.,1.]   ,  #[stddev_x, stddev_y] for gaussian
                 corr  = 0.        ,  #xy correlation
                 *args             ,
                 **kwargs          ):
  '''
  Define an initial wavefunction (Gaussian) with vortices!
  SYNTAX:
  def vortexgauss( x              ,  #xvals
                   y              ,  #yvals
                   vort           ,  #X by 3. array of vortex locations
                   means = [0.,0.],  #[x,y] centre of gaussian
                   sig   = [1.,1.],  #[stddev_x, stddev_y] for gaussian
                   corr  = 0.     ): #xy correlation
                   
  vort should look like:
  [ [-1,  0, +1],
    [-1, -1, -2],
    [+2, +3, +1],
    [ x,  y, ax] ] where ax specifies a vortex or antivortex and strength
  
  '''
  
  if abs( corr ) >= 1:
    print '<twod.vortexgauss> corr must be strictly between -1 and 1'
    
  
  
  #Gaussian Parameters and preliminary calculations for speed
  xsig = sig[0]
  ysig = sig[1]
  xycorr = np.array( [ [ xsig**2.,           corr * xsig * ysig ],
                      [ corr * xsig * ysig, ysig**2.           ] ] )
  
  corrdet = det(xycorr)
  
  xycorrinv = 1. / corrdet * np.array( [ [  xycorr[1,1], -xycorr[0,1]  ],
                                         [ -xycorr[1,0],  xycorr[0,0]  ] ] )
              
  corrdetroot = np.sqrt(corrdet)
  
  print '<twod.vortexgauss> combobulating initial wavefunction matrix'
  
  #Caclulate phase matrix
  X, Y = np.meshgrid( x, y )
  theta = np.zeros( np.shape( X ), dtype = complex )
  for X0, Y0, ax in vort: theta += ax * 1j * np.arctan2( ( Y - Y0) , ( X - X0) )
  
  #plt.figure("Initial Phase")
  #plt.contourf(x,y,np.angle(np.exp(theta)),20)
  #plt.xlabel(r'x ($a_0$)')
  #plt.ylabel(r'y ($a_0$)')
  #plt.show()
  
  wave = np.zeros( np.shape( X ), dtype = complex )
  #Now do the whole wavefunction
  for jj in range( 0, len(x) ):
    for kk in range( 0, len(y) ):
      pos = np.array( [ x[jj], y[kk] ] ) - means
      wave[jj,kk] = 1. / (2. * np.pi * corrdetroot ) * (
         np.exp( -0.5 * np.dot( np.dot( pos.transpose(), xycorrinv ), pos ) 
                ) * np.exp( theta[jj,kk] ) )
  print '<twod.vortexgauss> completed'
  
  return wave
  