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
#  ------------>rotate.py
#               Jan 24, 2013: Forked from twod.py
#                             Beginning conversion to RK4IP Method of
#                             Penckwitt (2004, PhD Thesis, Uni. of Otago, NZ)
#                             
#                             
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

from trapsave import *
from initials import *
from trapplot import *

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
    self.rad = np.sqrt( self.X ** 2. + self.Y ** 2. )
    
    self.psi = init( self.x, self.y, g = self.g, G = self.G, **kwargs )
    self.normalise()
    
    self.ksquare , self.k = self.k2dimen()
    self.V                = self.harm_trap()
    self.ker              = self.kernel()
    self.expksquare       = np.exp( -1.j * self.dt * self.ksquare )
    self.log              = np.log( self.rad )
    self.Lz               = self.Lzmake( **kwargs )
    
  #-----------------------------------------------------------------------------
  def normalise( self ):
    '''
    normalise psi
    call normalise() to normalise the wavefunction
    return is None
    '''
    self.psi = self.psi / np.sqrt( sum( sum ( abs( self.psi ) ** 2. ) )
             * self.dx * self.dy )
    return None
  #-----------------------------------------------------------------------------
  def k_vector( self ):
    '''Usual FFT format k vector (as in Numerical Recipes)'''
    delta = float( self.xmax - self.xmin ) / float( self.npt - 1 )
    k_pos = np.arange(      0, self.npt/2+1, dtype = float)
    k_neg = np.arange( -self.npt/2+1,     0, dtype = float)
    
    return (2*np.pi/self.npt/delta) * np.concatenate( (k_pos,k_neg) )

  #-----------------------------------------------------------------------------
  def k2dimen( self ):
    '''
    2d fft k-array, returns k^2, k both in 2d
    '''
    k_vec = self.k_vector()
    KX, KY = np.meshgrid ( k_vec, k_vec )
    k2squared = KX ** 2. + KY ** 2.
    k2 = np.sqrt( k2squared )
    k2[ self.npt/2:self.npt, 0:self.npt/2   ] = \
                                      -k2[ self.npt/2:self.npt, 0:self.npt/2   ]
    k2[ 0:self.npt/2  , self.npt/2:self.npt ] = \
                                      -k2[ 0:self.npt/2  , self.npt/2:self.npt ]
    
    return k2squared, k2

  #-----------------------------------------------------------------------------
  def harm_trap( self ):
    '''Harmonic potential,return 0.5 * self.P * self.rad ** 2.'''
    return 0.5 * self.P * self.rad ** 2.

  #-----------------------------------------------------------------------------
  def kernel( self ):
    '''
    Define the grid 1/abs(r)
    '''
    return abs( 1. / self.rad )

  #-----------------------------------------------------------------------------

  def gravity( self ):
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
  def stepRK4(self):
    '''
    Perform a timestep using the RK4 Interaction Picture
    This routine also considers real/imaginary parts
    '''
    n2 = self.npt ** 2.
    sq = [ self.npt, self.npt ]
    
    psii   = self.psi
    
    #Linear operator parts
    psiI = ff.fftshift( ff.ifft2( self.expksquare ** 0.5 * 
                         ff.fft2( ff.fftshift( self.psi ) ) ) ).reshape( n2 )
    
    #Nonlinear operator parts
    grav = spdiags( +1.j * self.gravity().reshape( n2 ), 0, n2, n2 )
    harm = spdiags( -1.j * self.V.reshape( n2 ), 0, n2, n2 )
    sint = spdiags( -1.j * self.g * abs( self.psi.reshape( n2 ) ) ** 2., 0, n2, n2 )
    
    nonlin = self.Lz + grav + harm + sint 
    
    #timesteps - note, we need to keep the pointer to self.psi the same
    #as this is requried if FFTW is ever implemented. We still need to update
    #self.psi at each step though to calculate self.gravity()
    ############################################################################
    k1 = ( self.dt * nonlin.dot( psiI ) ).reshape( sq )
    self.psi += ( k1 / 2. )
    
    grav = spdiags( +1.j * self.gravity().reshape( n2 ), 0, n2, n2 )
    sint = spdiags( -1.j * self.g * abs( self.psi.reshape( n2 ) ) ** 2., 0, n2, n2 )
    nonlin = self.Lz + grav + harm + sint 
    ############################################################################
    k2 = ( self.dt * nonlin.dot( psiI + k1.reshape( n2 ) / 2. ) ).reshape( sq )
    self.psi[:] = psii + k2 / 2.
    
    grav = spdiags( +1.j * self.gravity().reshape( n2 ), 0, n2, n2 )
    sint = spdiags( -1.j * self.g * abs( self.psi.reshape( n2 ) ) ** 2., 0, n2, n2 )
    nonlin = self.Lz + grav + harm + sint 
    ############################################################################
    k3 = ( self.dt * nonlin.dot( psiI + k2.reshape( n2 ) / 2. ) ).reshape( sq )
    self.psi[:] = psii + k3
    
    grav = spdiags( +1.j * self.gravity().reshape( n2 ), 0, n2, n2 )
    sint = spdiags( -1.j * self.g * abs( self.psi.reshape( n2 ) ) ** 2., 0, n2, n2 )
    nonlin = self.Lz + grav + harm + sint 
    ############################################################################
    psiII = ff.fftshift( ff.ifft2( self.expksquare * 
            ff.fft2( ff.fftshift( psiI.reshape( sq ) + k3 ) ) ) ).reshape( n2 )
    
    k4 = ( self.dt * nonlin.dot( psiII ) ).reshape( sq )
    
    ############################################################################
    
    
    self.psi[:] = ff.fftshift( ff.ifft2( self.expksquare ** ( self.dt / 2) * 
                  ff.fft2( ff.fftshift( psiI.reshape(sq) + 
                                        (k1 + 2 * ( k2 +k3 ) + k4 ) / 6. ) ) ) )
    if self.wick == True: self.normalise()
    return None
    
  #-----------------------------------------------------------------------------
  def wickon(self):
    '''
    simple function to ensure imaginary time propagation
    '''
    self.dt               = -1j * abs( self.dt )
    self.expksquare       = np.exp( -1.j * self.dt * self.ksquare )
    self.wick             = True #internal tracker of current wick state
    return None
  #-----------------------------------------------------------------------------
  
  def wickoff(self):
    '''
    simple function to ensure real time propagation
    '''
    self.dt               = abs( self.dt )
    self.expksquare       = np.exp( -1.j * self.dt * self.ksquare )
    self.wick             = False #internal tracker of current wick state
    return None
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
  
  def converge( self            ,
                tol             ,
                verbose = False ):
    '''
    def converge( self            ,
                  tol             , #tolerance
                  verbose = False ):
    A routine to check if the ground state has been reached.
    Returns true if virial <= tol (tolerance.
    Verbose will print the virial.
    virial = 'Ek - Ev + Ei - G/4'
    '''
    enList = self.energies( verbose = False )
    virial = np.real( enList[3] - enList[0] + enList[1] - self.G/4. )
    if verbose == True: print 'Ek - Ev + Ei - G/4 = ', virial
    return virial <= tol
  
  def Lzmake( self       ,
              order = 3  ,
              **kwargs   ):
    '''
    def Lzmake( self       ,
                order = 3  ,
                **kwargs   ):
    A routine to return Lz, the angular momentum finite difference matrix.
    Derivatives are from the PhD thesis of Andreas Penckwitt, PhD Thesis 2004
    Rotating Bose Einstein Condensates: Vortex Lattices and Excitations"
    www.physics.otago.ac.nz/nx/jdc/jdc-thesis-page.html"
    He also explains the RK4IP timestepping method
    Note that we worry about imaginary/real values in the timestepping
    '''
    assert order in [ 3, 5, 7, 9, 11, 13 ],\
      'Only order values contained in [ 3, 5, 7, 9, 11, 13 ] are implemented\
     \nthus far, please ensure you are have the order type correct also.'
    
    if order == 3:
      a = np.array( [ 0., 0., 0., 0., 0., -1., 0., 1., 0., 0., 0., 0., 0. ] )
      denom = 2.
      a /= denom
    elif order == 5:
      a = np.array( [ 0., 0., 0., 0., 1., -8., 0., 8., -1., 0., 0., 0., 0. ] )
      denom = 12.
      a /= denom
    elif order == 7:
      a = np.array( [ 0., 0., 0., -1., 9., -45., 0., 45., -9., 1., 0., 0., 0. ] )
      denom = 60.
      a /= denom
    elif order == 9:
      a = np.array( [ 0., 0., 3., -32., 168., -672., 0., 672., -168., 32., -3., 0., 0. ] )
      denom = 840.
      a /= denom
    elif order == 11:
      a = np.array( [ 0., -2., 25., -150., 600., -2100., 0.,
                      2100., -600., 150., -25., 2., 0. ] )
      denom = 2520.
      a /= denom
    elif order == 13:
      a = np.array( [ 5., -72., 495., -2200., 7425., -23760., 0.,
                      23760., -7425., 2200., -495., 72., -5. ] )
      denom = 27720.
      a /= denom
    
    elif type(order) != int: return SystemError('Order must be type int')
    else: return SystemError('Invalid FEM order: order = 3, 5, 7, 9, 11 or 13')
    
    data = []
    diag = []
    for jj in np.where( [ii != 0 for ii in a] )[0]:
      #Iterate over the diagonals in a that are non-zero
      #offset of jj from array centred
      off = jj - np.ceil( jj/2. )
      data.append( a[jj] * np.ones( [ 1, self.npt ** 2 ] ) )
      diag.append( off )
      
      data.append( a[jj] * np.ones( [ 1, self.npt ** 2 ] ) )
      diag.append( self.npt + off )
    
    data = np.vstack(data) #put the data in rank 2 array
    
    return self.rot * spdiags( data, diag, self.npt ** 2., self.npt ** 2. )
  
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
