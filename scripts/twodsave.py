################################################################################
#
# Program:      2D BEC simulation with gravitational potential 
#
# Author:       Nathanael Lampe, School of Physics, Monash University
#
# Created:      December 4, 2012 (Forked from twodtrap.py)
#
# Changelog:    twodtrap.py
#               Aug 29, 2012: Created
#               Oct 11, 2012: Working, changelog begun
#               Oct 11, 2012: Implementation of rotation into Hamiltonian
#               twod.py
#               Dec 04, 2012: Created from twodtrap.py
#               Dec 12, 2012: Implemented scheme to check cnvergence against the
#                             chemical potential
#               Jan 24, 2012: The program works in accordance with the virial 
#                             theorem, and TF approximation, Has now been forked
#                             to rotate.py, where rotation will be implemented.
#           --->twodsave.py
#               Jan 30, 2013: Forked from twod.py
#                             This version will contain the save routine, the 
#                             other will just contain the Bose class and 
#                             related functions to avoind import confilicts
#               
#
# Purpose:      Provide a function to evaluate the stability of a 
#               gravitationally bound BEC under varying trap strengths
#               
################################################################################

#Imports:
from __future__ import division

import numpy as np
import scipy.fftpack as ff
import scipy.linalg as la

from scipy.sparse import spdiags
from scipy.special import j0, j1, jn_zeros

from traphdf5 import *
from initials import *
from trapplot import *

################################################################################

def k_vector( n, extent ):
  '''Usual FFT format k vector (as in Numerical Recipes)'''
  delta = float(extent) / float(n-1)
  k_pos = np.arange(      0, n/2+1, dtype = float)
  k_neg = np.arange( -n/2+1,     0, dtype = float)
  return (2*np.pi/n/delta) * np.concatenate( (k_pos,k_neg) )

################################################################################

def k2dimen(a,b,npt):
  k_vec = k_vector(npt, b - a )
  KX, KY = np.meshgrid ( k_vec, k_vec )
  k2squared = KX ** 2. + KY ** 2.
  k2 = np.sqrt( k2squared )
  
  #k_vec = k_vector(npt, b - a )
  #k2d = np.zeros([npt,npt])
  #kf2 = np.zeros([npt,npt])
  #for ii in range(0,npt):
    #for jj in range(0,npt):
      #kf2[ii,jj] = k_vec[ii]**2+k_vec[jj]**2
      #k2d[ii,jj] = np.sqrt(kf2[ii,jj])
  k2[ npt/2:npt, 0:npt/2   ] = -k2[ npt/2:npt, 0:npt/2   ]
  k2[ 0:npt/2  , npt/2:npt ] = -k2[ 0:npt/2  , npt/2:npt ]
  return k2squared, k2

################################################################################



################################################################################

def harm_trap( x, y, P ):
  X, Y = np.meshgrid(x,y)
  return 0.5 * P * ( X**2 + Y**2 )

################################################################################
  
def kernel( x    , 
            y    ,
            npt  ):
  '''
  Define the grid 1/abs(r)
  '''
  ker = np.zeros([npt,npt])
  for ii in range(0,npt):
    for jj in range(0,npt):
      ker[ii,jj] = abs(1./np.sqrt(x[ii]**2.+y[jj]**2.))
  return ker

################################################################################
  
class Bose:
  '''
  Establish the Bose class
  The Bose class defines an item that contains most relevant arrays
  that are passed to functions. Many of these are to optimise the routine
  '''
  def __init__( self, a, b, npt, init, g, G, rot, P, dt, **kwargs ):
    self.xmin = a
    self.xmax = b
    self.npt  = npt
    self.P    = P
    self.g    = g
    self.G    = G
    self.rot  = rot
    self.dt   = dt
    
    self.x  = np.linspace( self.xmin, self.xmax, npt )
    self.dx = abs( self.x[1] - self.x[0] )
    self.y  = self.x
    self.dy = abs( self.y[1] - self.y[0] )
    self.X, self.Y = np.meshgrid(self.x, self.y)
    
    self.psi              = init      ( self.x, self.y, self.npt, **kwargs )
    self.psi              = self.psi / np.sqrt( sum(
                                                   sum ( abs( self.psi ) ** 2. )
                                                   ) * self.dx * self.dy )
                          #This ensures the wavefunction is normalised
    
    self.ksquare , self.k = k2dimen   ( self.xmin, self.xmax, self.npt )
    self.V                = harm_trap ( self.x, self.y, self.P )
    self.ker              = kernel    ( self.x, self.y, self.npt )
    
    self.expksquare       = np.exp    ( -0.5j * self.dt * self.ksquare )
    
    self.log              = np.log( np.sqrt( self.X ** 2. + self.Y ** 2. ) )
    
    #self.Lz               = self.Lzmake( order = 2. )
    
  #-----------------------------------------------------------------------------
  
  def gravity(self):
    '''
    Evaluate the gravitational field, with a call to Bose.gravity()
    Gravitaional field is the convolution of the density and the log of distance
    '''
    den = abs(self.psi)**2.  #calculate the probability density
    
    #return the convolution, after multiplying by scaled gravity and 
    #correcting for grid scaling (due to 2 forward FFTs, and only one inverse
    return self.G * self.dx * self.dy * (
           ff.fftshift( ff.ifft2( ff.fft2( ff.fftshift( den ) ) * 
                       abs( ff.fft2( ff.fftshift( -self.log ) ) ) ) 
                      )                  )
  
  #-----------------------------------------------------------------------------
  def fgravity( self            ,
                verbose = False ):
    '''
    fgravity( self           ,
              verbose =False ):
    Evaluate the gravitational field by FEM, with a call to Bose.gravity()
    Gravitaional field is the convolution of the density and the kernel
    Uses self.ker = 1/|r|, self.psi = wavefunction
    
    verbose = True will plot both the FEM gravity and FFT gravity
    '''
    den = abs(self.psi)**2.  #calculate the probability density
    
    #return the convolution, after multiplying by scaled gravity and 
    #correcting for grid scaling (due to 2 forward FFTs, and only one inverse
    grav = np.zeros( [ self.npt, self.npt ] )
    #for ii in range( 1, self.npt - 1 ):
      #for jj in range( 1, self.npt - 1 ):
        #integral = 0.
        #X, Y = np.meshgrid( self.x, self.y )
        #grav[ii,jj] = self.dx * self.dy * sum( sum( den / np.sqrt(
                     #( X - self.x[ii] ) ** 2. + ( Y - self.y[jj] ) ** 2. ) ) )
        #for kk in range( 1, self.npt - 1 ):
          #for ll in range( 1, self.npt - 1 ):
            #if kk != ii and ll != jj: integral += den[kk, ll] / np.sqrt(
                                      #( self.x[kk] - self.x[ii] ) ** 2. + 
                                      #( self.y[ll] - self.y[jj] ) ** 2. )
        #grav[ii,jj] = integral * self.dx * self.dy
    for ii in range(1, self.npt -1):
      y = self.y[self.npt//2]
      integral = 0
      for kk in range( 1, self.npt - 1):
        for ll in range( 1, self.npt - 1):
          if kk != ii and ll != self.npt//2:
            integral += den[kk, ll] / np.sqrt(
                                    ( self.x[kk] - self.x[ii] ) ** 2. + 
                                    ( self.y[ll] - y) ** 2. )

      grav[self.npt//2,ii] = self.G * integral * self.dx * self.dy
  
    if verbose == True:
      gfig = plt.figure()
      gax = gfig.add_subplot(111)
      gax.plot(self.x, grav[ self.npt//2, : ], label = 'FEM')
      gax.plot(self.x, self.gravity()[ self.npt//2, : ],
                                                label = 'FFT' )
      lgd = plt.legend( loc = 'upper right' )
      plt.show()
    
    return grav
  
  #-----------------------------------------------------------------------------
  
  
  def angular(self):
    '''
    return the angular momentum operator value at each point in the grid
    L = -i ( x * d/dy - y * d/dx ) * psi
    We assume a square grid centred on  (x,y) = (0,0)
    '''
    kx = k_vector( self.npt, self.xmax - self.xmin )
    ky = kx
    
    KX, KY = np.meshgrid( kx, ky )
    X ,  Y = np.meshgrid( self.x, self.y )    
    fpsi   = ff.fft2( ff.fftshift( self.psi ) )
    
    
    return self.rot *1j * ( X * ff.fftshift( ff.ifft2( -1j * KY * fpsi) ) -
                            Y * ff.fftshift( ff.ifft2( -1j * KX * fpsi) )
                          ) 
  #-----------------------------------------------------------------------------
  
  def step2(self):
    '''
    Perform a second order timestep
    '''
    
    V2 = np.exp( -1j * self.dt / 2. * (self.V - self.gravity() + 
                                       self.g * abs( self.psi ) ** 2
                                       ) 
                )
                   
    return V2 * ff.fftshift( ff.ifft2(self.expksquare * 
                                    ff.fft2( ff.fftshift( V2 * self.psi ) )
                                    )
                           )
  
  #-----------------------------------------------------------------------------
  
  def step4(self):
    '''
    Perform a 4th order timestep
    '''
    
    def order2(c):
      Vc = np.exp( -1j * c * self.dt / 2. * 
                    ( self.V - self.gravity() + 
                      self.g * abs( self.psi ) ** 2
                    )
                  )
      Tc = self.expksquare ** c
      return Vc, Tc
    
    p = 1/(4.-4.**(1/3.))
    q = 1 - 4 * p
    
    Vp,Tp = order2( p )
    Vq,Tq = order2( q )
    
    return Vp * ff.fftshift( ff.ifft2( Tp * ff.fft2( ff.fftshift( Vp ** 2 * 
                ff.fftshift( ff.ifft2( Tp * ff.fft2( ff.fftshift( Vp * Vq *
                ff.fftshift( ff.ifft2( Tq * ff.fft2( ff.fftshift( Vq * Vp * 
                ff.fftshift( ff.ifft2( Tp * ff.fft2( ff.fftshift( Vp ** 2 *  
                ff.fftshift( ff.ifft2( Tp * ff.fft2( ff.fftshift( Vp * self.psi 
                ) ) ) )
                ) ) ) )
                ) ) ) )
                ) ) ) )
                ) ) ) )
    
  #-----------------------------------------------------------------------------
  
  #-----------------------------------------------------------------------------
  
  def IPStep(self):
    '''
    Perform a timestep using the RK4 Interaction
    '''
    return None
  #-----------------------------------------------------------------------------
  def wickon(self):
    '''
    simple function to ensure imaginary time propagation
    '''
    self.dt               = -1j * abs(self.dt)
    self.expksquare       = np.exp( -0.5j * self.dt * self.ksquare )
  #-----------------------------------------------------------------------------
  
  def wickoff(self):
    '''
    simple function to ensure real time propagation
    '''
    self.dt               = abs(self.dt)
    self.expksquare       = np.exp( -0.5j * self.dt * self.ksquare )
  #-----------------------------------------------------------------------------
  
  def energies( self            ,
                verbose = False ):
    '''
    energies(self, verbose = False)
    bec is an instance of the Bose class
    Returns the list of energies enList = [ Ev, Ei, Eg, Ekin ]
    This is the harmonic, interaction, gravitational and kinetic energies
    '''
    
    
    Ev = sum( sum( self.psi.conjugate() * self.V * self.psi )
            ) * self.dx * self.dy
    
    Ei = sum( sum( self.psi.conjugate() * 0.5 * self.g * abs(self.psi) **2. *
                   self.psi ) ) * self.dx * self.dy
    
    Eg = sum( sum( self.psi.conjugate() * -1. * self.gravity() * self.psi )
            ) * self.dx * self.dy
    
    Ekin = sum( 
               sum(
        self.psi.conjugate() * ff.fftshift( ff.ifft2( 0.5 * self.ksquare * 
                                            ff.fft2( ff.fftshift( self.psi ) ) ) 
                                          )
                   )
              ) * self.dx * self.dy
    
    enList = [ Ev, Ei, Eg, Ekin ]
    
    
    
    if verbose == True: #print out the energies
      #Calculate gravitational field Laplacian
      gf = -1. * self.gravity() #self.gravity() is +ve
      glpg = ( np.roll( gf, 1, axis = 0 )
             + np.roll( gf,-1, axis = 0 )
             + np.roll( gf, 1, axis = 1 )
             + np.roll( gf,-1, axis = 1 )
             - 4 * gf ) / self.dx ** 2.
      glpd = 2. * np.pi * self.G * abs(self.psi) ** 2.
      
      ##diagnostic plots
      #gravfig = plt.figure()
      #gravax = gravfig.add_subplot(111)
      #gravax.plot(self.x, glpg[ self.npt//2, : ], label = 'Fourier Laplace' )
      #gravax.plot(self.x, glpd[ self.npt//2, : ], label = 'Density Laplace' )
      #gravlgd = plt.legend(loc='upper right')
      #plt.show()
      
      print 'Harmonic PE         = ', np.real( enList[0] )
      print 'Interaction PE      = ', np.real( enList[1] )
      print 'Gravitational PE    = ', np.real( enList[2] )
      print 'Potential Energy    = ', np.real( sum( enList[0:3] ) )
      print 'Kinetic Energy      = ', np.real( enList[3] )
      print 'Total Energy        = ', np.real( sum( enList ) )
      print 'normalisation       = ', np.real( sum( sum( abs( self.psi ) ** 2. ) 
                                                  ) * self.dx * self.dy )
      print 'Chemical Potential  = ', np.real( enList[3] + enList[0] + 
                                               2 * enList[1] + enList[2] )
      print 'Ek - Ev + Ei - G/4  = ', np.real( enList[3] - enList[0] + enList[1]
                                             - self.G/4. )
      print 'Laplace Eq check    = ', np.real( sum( sum( glpg - glpd ) ) )
    return enList
  #-----------------------------------------------------------------------------
  
  def TFError( self,
               verbose = False):
    '''
    TFError( self, verbose = False):
    A function to evaluate the deviation of the solution from the Thomas-Fermi
    approximation.
    Setting verbose to true will produce plots of the wavefunction and TF 
    approximation
    '''
    
    enList = self.energies( verbose = False )
    #mu    = <K>       + <Vext>    + 2 * <Vsi>     + 0.5 * <Vgrav>
    fineX = np.arange(self.xmin , self.xmax, (self.xmax - self.xmin) / 
                                             ( 1e3 * self.npt ), dtype = float )
    fineY = fineX
    X, Y  = np.meshgrid( self.x, self.y )
    Rsq   = X ** 2. + Y ** 2.
    R     = np.sqrt( Rsq )
    
    if self.g != 0 and self.G == 0. and self.P != 0.:
      #Harmonic case
      chmpot = enList[3] + enList[0] + 2 * enList[1]
      
      r0sq   = 2 * chmpot / self.P
      tfsol  = (chmpot / self.g - Rsq * self.P / ( 2 * self.g ) ) * ( 
                                          map( lambda rsq: rsq - r0sq < 0, Rsq )
                                               )
      tferror = np.real( sum( sum( tfsol - abs( self.psi ) ** 2. ) ) )
      print 'Thomas-Fermi Error = ', tferror
      print 'TF Error per cell  = ', tferror / self.npt ** 2.
      
      if verbose == True:
        #diagnostic plots
        
        tfx  = (chmpot / self.g - fineX ** 2. * self.P / ( 2 * self.g ) ) * ( 
                                     map( lambda fineX: fineX **2. - r0sq < 0,
                                          fineX )                           )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(fineX, tfx, label = 'analytic solution')
        
        ax.plot(self.x, abs( self.psi[ self.npt//2, : ] ) ** 2.,
                                                  label = 'numerical solution' )
        
        lgd = plt.legend( loc = 'upper right' )
        plt.show()
      
      
    elif self.G != 0. and self.g != 0. and self.P == 0.:
      #Gravitational case
      bj0z1   = jn_zeros( 0, 1 ) #First zero of zeroth order besselj
      scaling = np.sqrt( 2 * np.pi * self.G / self.g  )
      gr0     = bj0z1 / scaling
      Rprime  = R * scaling
      
      gtfsol = j0( Rprime ) * np.array( [ map( int,ii ) for ii in map( 
                                         lambda rad: rad <= gr0, R ) ] )

      
      gtfsol *= scaling ** 2. / ( 2 * np.pi * j1( bj0z1 ) * bj0z1 ) #(abs(self.psi) ** 2.).max()
      
      #gtfsol = 1. / ( 4. * gr0 ** 2. * ( 1 - 2 / np.pi ) ) * np.cos(
        #np.pi * R / ( 2. * gr0 ) ) * map( lambda rsq: rsq - gr0 ** 2. < 0, Rsq )
      gtferror = np.real( sum( sum( gtfsol - abs( self.psi ) ** 2. ) ) ) 
      
      print 'Grav. TF Error     = ', gtferror
      print 'GTF Error per cell = ', gtferror / self.npt ** 2.
      print 'Analytic norm      = ', sum( sum( gtfsol ) ) * self.dx * self.dy
      
      if verbose == True:
        #diagnostic energies
        gtfwf = np.sqrt( gtfsol )
        
        Ev = 0.
        
        Ei = sum( sum( 0.5 * self.g * abs(gtfwf) ** 4. ) 
                ) * self.dx * self.dy
        
        GField = self.G * self.dx * self.dy * (
                 ff.fftshift( ff.ifft2( ff.fft2( ff.fftshift( gtfsol ) ) * 
                       abs( ff.fft2( ff.fftshift( -self.log ) ) ) ) 
                            )                 )
        Eg = sum( sum( -1. * GField * gtfsol ) ) * self.dx * self.dy
        
        Ekin = sum( sum( gtfwf.conjugate() * 
                         ff.fftshift( ff.ifft2( 0.5 * self.ksquare * 
                         ff.fft2( ff.fftshift( gtfwf ) ) ) 
                                     )
                       )
                  ) * self.dx * self.dy
        
        TFList = [ Ev, Ei, Eg, Ekin ]
        
        #glpg = - 1. * ( np.roll( GField, 1, axis = 0 ) #GField is +ve
                      #+ np.roll( GField,-1, axis = 0 )
                      #+ np.roll( GField, 1, axis = 1 )
                      #+ np.roll( GField,-1, axis = 1 )
                      #- 4 * GField ) / self.dx ** 2.
        #glpd = 2 * np.pi * self.G * gtfsol 
        
        print '\nTF solution Energies\n'
        print 'Harmonic PE         = ', np.real( TFList[0] )
        print 'Interaction PE      = ', np.real( TFList[1] )
        print 'Gravitational PE    = ', np.real( TFList[2] )
        print 'Potential Energy    = ', np.real( sum( TFList[0:3] ) )
        print 'Kinetic Energy      = ', np.real( TFList[3] )
        print 'Total Energy        = ', np.real( sum( TFList ) )
        print 'Chemical Potential  = ', np.real( TFList[3] + TFList[0] + 
                                                2 * TFList[1] + TFList[2] )
        print 'Ek - Ev + Ei - G/4  = ', np.real( TFList[3] - TFList[0] +
                                                  TFList[1] - self.G / 4. )
        
        #diagnostic plots
        fineXS = fineX * scaling
        gtx = j0( fineXS ) * np.where( abs(fineX) < gr0,
                              np.ones( len(fineX) ), np.zeros( len(fineX) ) )
        gtx *= scaling ** 2. / ( 2 * np.pi * j1( bj0z1 ) * bj0z1 )
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(fineX, gtx, label = 'analytic solution')
        ax.plot(self.x, abs( self.psi[ self.npt//2, : ] ) ** 2.,
                                                  label = 'numerical solution' )
        
        lgd = plt.legend( loc = 'upper right' )
        plt.show()
      
    else:
      print 'A TF approximation for this scenario has not yet been implemented'
      print 'Sorry about that...'
      
################################################################################
  
  def virial( self            ,
              verbose = False ):
    '''
    def converge( self            ,
                  verbose = False ):
    A routine to check if the ground state has been reached.
    Returns the virial
    Verbose will print the virial.
    virial = 'Ek - Ev + Ei - G/4'
    '''
    enList = self.energies( verbose = False )
    virial = np.real( enList[3] - enList[0] + enList[1] - self.G/4. )
    if verbose == True: print 'Ek - Ev + Ei - G/4 = ', virial
    return virial
    
  def Lzmake( self       ,
              order = 2. ):
    '''
    A routine to return Lz, the angular momentum finite difference matrix.
    Derivatives are from:
    Generation of finite difference formulas on arbitrarily spaced grids
    Fornberg, B.,Generation of finite difference formulas on arbitrarily spaced
    grids, Math. Comp. 51 (1988), 699-706 
    '''
    
    if order == 2:
      a = [ 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0 ]
      denom = 2.
      a /= denom
    elif order == 5:
      a = [ 0, 0, 0, 0, 1, -8, 0, 8, -1, 0, 0, 0, 0 ]
      denom = 12.
      a /= denom
    elif order == 7:
      a = [ 0, 0, 0, -1, 9, -45, 0, 45, -9, 1, 0, 0, 0 ]
      denom = 60.
      a /= denom
    elif order == 9:
      a = [ 0, 0, 3, -32, 168, -672, 0, 672, -168, 32, -3, 0, 0 ]
      denom = 840.
      a /= denom
    elif order == 11:
      a = [ 0, -2, 25, -150, 600, -2100, 0, 2100, -600, 150, -25, 2, 0 ]
      denom = 2520.
      a /= denom
    elif order == 13:
      a = [ 5, -72, 495, -2200, 7425, -23760, 0,
            23760, -7425, 2200, -495, 72, -5 ]
      denom = 27720.
      a /= denom
    
    elif type(order) != int: print 'Order must be type int'
    else: print 'ensure order is 3, 5, 7 or 9'
    
    data = []
    diag = []
    for jj in where([ii != 0 for ii in a])[0]:
      #Iterate over the diagonals in a that are non-zero
      #offset of jj from array centred
      off = jj - np.ceil( jj/2. )
      data.append( a[jj] * np.ones( [self.npt ** 2, 1] ) )
      diag.append( off )
      
      data.append( a[jj] * np.ones( [self.npt ** 2, 1] ) )
      diag.append( self.npt + off )
    
    Lz = spdiags( data, diag, self.npt ** 2., self.npt ** 2. )
    
    
    return Lz #we assume dx=dy
  
################################################################################
################################################################################

def twodsave( a                      ,  #min co-ord
              b                      ,  #max co-ord
              npt                    ,  #no. gridpoints
              dt                     ,  #timestep
              tstop                  ,  #time to stop
              g                      ,  #s-wave scattering strength
              G                      ,  #gravitational field scaled strength
              rot                    ,  #angular momentum in system
              filename = autof       ,  #output filename, autof for automatic naming
              P        = 0.          ,  #optional harmonic potential
              wick     = False       ,  #Wick rotation True/False
              init     = vortexgauss ,  #initial wavefunction shape (function call)
              skip     = 1.          ,  #intervals to skip when saving
              energies = False       ,  #print initial and final energy
              erase    = False       ,  #overwrite existing files True/False
              **kwargs                ):
  '''
  save the result of a gravitational simulation of a BEC
  saved file is a hd5 database
  SYNTAX:
    def twodsave( a                      ,  #min co-ord
                  b                      ,  #max co-ord
                  npt                    ,  #no. gridpoints
                  dt                     ,  #timestep
                  tstop                  ,  #time to stop
                  g                      ,  #s-wave scattering strength
                  G                      ,  #gravitational field scaled strength
                  rot                    ,  #angular momentum in system
                  filename = autof       ,  #output filename, autof for 
                                            #automatic naming
                  P        = 0.          ,  #optional harmonic potential
                  wick     = False       ,  #Wick rotation True/False
                  init     = vortexgauss ,  #initial wavefunction shape 
                                            #(function call)
                  skip     = 1.          ,  #intervals to skip when saving
                  energies = False       ,  #print initial and final energy
                  erase    = False       ,  #overwrite existing files True/False
                  **kwargs                ):
  '''
  #initial setup ---------------------------------------------------------------
  
  #Prepare parameters for database
  h = dict( {'G'        : G                    ,
             'g'        : g                    ,
             'rot'      : rot                  ,
             'P'        : P                    ,
             'wick'     : wick                 ,
             'dt'       : dt                   ,
             'tstop'    : tstop                ,
             'xmin'     : a                    ,
             'xmax'     : b                    ,
             'npt'      : npt                  ,
             'skipstep' : skip                 ,
             'steps'    : (tstop // dt) // skip } )
  if init == vortexgauss:
    try: h['vortices'] = kwargs['vort']
    except:
      print 'function vortexgauss requires specification of vortex locations'
      print 'use "print twodtrap.vortexgauss.__doc__" for correct syntax'
      return SystemError('Aborting as could not find kwarg "vort"')
  else: h['vortices'] = ''
  
  if filename == autof: #automatically add run to database, and name file
    filename = autof(h)
  else: #custom filename
    if type(filename) != str: return SystemError('Filename should be a string')
  
  #Make a condensate
  bec = Bose( a, b, int(npt), init, g, G, rot, P, dt, **kwargs)  
  if energies == True:
    print 'Initial Energies\n'
    bec.energies( verbose = True )
  
  if wick == True: #Enable Wick Rotation
    bec.wickon()  
  
  if wick == False: #propogate for a brief time in wick space to remove 
                    #numerical vortex artifacts from the simulation
    bec.dt = dt/100. #MUST GO BEFORE wickon()
    bec.wickon()
    
    bec.step4()
    bec.step4()
    bec.step4()
    
    bec.dt = dt   #MUST GO BEFORE wickoff
    bec.wickoff()
  
  infile = h5file(filename, erase = erase, read = False )
  infile.add_headers( h )
  infile.add_data(str('0.0'), bec.x, bec.y, bec.psi, 0)
  
  
  #normalise the wavefunction
  norm      = sum(sum( bec.psi * ( bec.psi ).conjugate() )) * bec.dx * bec.dy
  bec.psi   = bec.psi / np.sqrt( norm )
  #wavefunction normalised so probability = 1  
  

  
  
  savecounter = 0.
  saves       = 0.
  # time-evolution--------------------------------------------------------------
  for t in np.arange(0,tstop,dt):
    bec.psi = bec.step4()
    savecounter += 1.
    
    if wick == True:  #normalise after Wick rotation
      norm  = sum( sum( abs( bec.psi )**2 ) ) * bec.dx * bec.dy
      bec.psi = bec.psi / np.sqrt(norm)
      
    if savecounter == skip: #save the data
      infile.add_data(str( saves + 1.), bec.x, bec.y, bec.psi, t)
      savecounter = 0.
      saves += 1.
      print t
  
  if energies == True:
    print 'Final Energies\n'
    bec.energies( verbose = True )
    print '\n Thomas-Fermi Accuracy\n'
    bec.TFError( verbose = False )
    #bec.fgravity( verbose = True )
  
  print 'Run saved to ' + filename
  print 'Parameters added to sims/runs.info'
  print 'Have a nice day :)'
  infile.f.close()

################################################################################
def twod( a                 ,
          b                 ,
          npt               ,
          dt                ,
          tstop             ,
          g                 ,
          G                 ,
          rot               ,
          P        = 0.     ,
          wick     = False  ,
          init     = gauss  ,
          analysis = False  ,
          **kwargs          ):
  '''
  Return the result of a gravitational simulation of a BEC
  xvals, tvals, psi = oned(a,b,npt,dt,tstop,g,G,P=0.,wick=False):
  a = xmin
  b = xmax
  npt = spatial steps
  dt = timestep size
  tstop = stopping time
  g = s-wave self-interaction strenth
  G = gravitational interaction strength
  P = harmonic potential strength (default 0)
  '''
  
  bec = Bose( a, b, int(npt), init, g, G, rot, P, dt, **kwargs)
  
  if wick == True : bec.wickon()  #Wick Rotation
  
  #normalise the wavefunction
  norm      = sum(sum( bec.psi * ( bec.psi ).conjugate() )) * bec.dx * bec.dy
  bec.psi   = bec.psi / np.sqrt( norm )
  #wavefunction normalised so probability = 1  
  
  print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
  print 'Initial Energies'
  Epot, Ekin = energies(bec)
  print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
  Einit = Epot+Ekin
  
  #prepare data output arrays
  gravout = [bec.gravity()]
  results = [bec.psi]
    
  jj = int(0)     #A simple counter
  stable = False  #Changes to true when the routine converges under
                  #Wick rotation, if the analysis function is active
  
  # time-evolution
  for t in np.arange(0,tstop,dt):
  
    bec.psi = bec.step4()
    
    if wick == True:
      norm  = sum( sum( abs( bec.psi )**2 ) ) * bec.dx * bec.dy
      bec.psi = bec.psi / np.sqrt(norm)
    
    results.append(bec.psi)
    gravout.append(bec.gravity())
    
    if jj == (100 * (jj // 100)):
      
      #print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
      #print 'Energies at', jj
      #oldE = Epot + Ekin
      #Epot, Ekin = energies(bec)
      #print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
      
      if wick == True and analysis == True and abs(Epot + Ekin - oldE) < 1e-4:
        stable = True
        break
    jj += 1
    
  print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
  print 'Final Energies'
  Epot, Ekin = energies(bec)
  print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
  
  return bec, np.arange(0,jj*dt,dt), results, gravout, stable