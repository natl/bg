########################################################################
#
# Program:      2D BEC simulation with gravitational potential 
#
# Author:       Nathanael Lampe, School of Physics, Monash University
#
# Created:      August 29, 2012
#
# Changelog:
#               October 11, 2012: Working, changelog begun
#               October 11, 2012: Implementation of rotation into Hamiltonian
#               December 4, 2012: Retired in favour of a multi-script approach
#                                 that separates different program functions
#                                 
#                                 Fixed a glitch in 4th order stepping due to 
#                                 not fftshifting the potential, this could
#                                 lead to a bug in gravity, but I'll be 
#                                 checking that in the new scripts
#           
#               
#
# Purpose:      Provide a function to evaluate the stability of a 
#               gravitationally bound BEC under varying trap strengths
#               
########################################################################
#Imports:
from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import fftw3 as fftw
import scipy.fftpack as f
import h5py
from numpy.linalg import det

def autof(h):
  '''
  Automatically get file database, make a filename, and add headers to a 
  database file
  
  headers are specified by h
  '''
  dfile = open('sims/runs.info', 'r+')
  
  run = len( dfile.readlines() ) - 2. #How many lines in the file?
                                      #run = file lines - 1 (due to
                                      #file structure.
  
  filename = ("sims/tdgb%i.hdf5" % run) #two-d gravitational bec X .hdf5
  
  v=[]
  for row in h[ 'vortices' ]: v.append( list( row ) ) #get vortices in 
                                                      #a nice format
  
  dfile.write( '%4i      %5s     %8f  %8f  %8f  %8f  %8f  %8f  %8f  %8f  %8f  %8f  %8f  %s\n' %
               ( run, h['wick'], h['G'], h['g'], h['rot'], h['P'], h['dt'], h['tstop'],
                 h['xmin'], h['xmax'], h['npt'], h['skipstep'], h['steps'],
                 str(v) )
              )
  
  
  return filename

def createh5( name          ,
              erase = False ):
  '''
  Open a hdf5 file, checking that nothing is accidentally overwritten
  '''
  if erase == False:
    #open file, check if we can overwrite previous files
    try:
        f = h5py.File( name, 'w-' )
    except:
      print 'File exists already'
      print 'add kwarg erase = True to overwrite'
      return SystemError('Aborting so you do not lose precious data')
      
  #Overwrite existing file if erase = True
  elif erase == True: f = h5py.File( name, 'w' )
    
  return f

def readh5( name ):
  '''
  Open a hdf5 file for reading
  '''
  try:
    f = h5py.File( str(name) , 'r')
  except:
    print "It seems the file you tried to open doesn't exist."
    print "sorry about that :("
  
  headers = dict()
  
  print "Succesfully opened" + str(name)
  print "File headers are as follows"
  
  for name, value in f.attrs.iteritems():
    print name+":", value
    headers[ name ] = value
  
  print 'file headers are stored in the dictionary h5file.head'
  
  return f, headers
  

class h5file:
  '''
  The overarching HDF5 file class for handling these files
  USAGE: 
    a = h5file( filename    ,    #filepath to open
                erase       ,    #overwrite existing file?
                read = False ):  #reading in a file instead? Then True
  METHODS:
    self.add_data
    self.add_headers
    self.readpsi
    self.readxy
  Each method has its own documentation.
  '''
  def __init__( self        ,
                filename    ,    #filepath to open
                erase       ,    #overwrite existing file?
                read = False ):  #reading in a file instead? Then True
    self.name   = filename
    self.erase  = erase
    
    if read == False: 
      self.f = createh5( self.name, erase = erase)
    elif read == True: self.f, self.head = readh5( filename )
  
  def add_data(self, runname, xdata, ydata, psi, time):
    
    grp   = self.f.create_group( str(runname) )
    grp.attrs['time'] = time
    
    xvals = grp.create_dataset( 'xvals', data = xdata )
    yvals = grp.create_dataset( 'yvals', data = ydata )
    psi   = grp.create_dataset( 'psi'  , data = psi   )
    
    
  def add_headers(self, head_dict):
    '''
    Add headers to the main file, to explain relevant parameters
    SYNTAX: h5file.add_headers(head_dict):
    head_dict should be a dictionary
    '''
    
    for name, value in head_dict.iteritems():
      self.f.attrs[ str( name ) ] = value
  
  def readxy(self):
    '''
    Will return the grid for the chosen run
    x,y = self.readxy() 
    '''
    return self.f.get('0.0/xvals')[:], self.f.get('0.0/yvals')[:]
    
  def readpsi(self, i):
    '''
    Will return a specified psi
    p = self.readpsi('frame')
    frame is an integer (within a string) in the form 0.0, 1.0, etc.
    '''
    return self.f.get( str(i) + '/psi')[:]

def lastpsi(x,psi):
  return psi

def k_vector( n, extent ):
  '''Usual FFT format k vector (as in Numerical Recipes)'''
  delta = float(extent) / float(n-1)
  k_pos = np.arange(      0, n/2+1, dtype = float)
  k_neg = np.arange( -n/2+1,     0, dtype = float)
  return (2*np.pi/n/delta) * np.concatenate( (k_pos,k_neg) )

def k2dimen(a,b,npt):
  k_vec = k_vector(npt, b - a )
  k2d = np.zeros([npt,npt])
  kf2 = np.zeros([npt,npt])
  for ii in range(0,npt):
    for jj in range(0,npt):
      kf2[ii,jj] = k_vec[ii]**2+k_vec[jj]**2
      k2d[ii,jj] = np.sqrt(kf2[ii,jj])
  k2d[npt/2:npt,0:npt/2] = -k2d[npt/2:npt,0:npt/2]
  k2d[0:npt/2,npt/2:npt] = -k2d[0:npt/2,npt/2:npt]
  return kf2,k2d

def energies(bec):
  '''
  Return the potential and kinetic energies, and print them
  Inputs are the wavefunction, potential, k vector and dx
  '''
  poten = bec.V + bec.g * abs(bec.psi)**2 - bec.gravity() + bec.angular()
  
  Epot = sum( sum( bec.psi.conjugate() * poten * bec.psi ) ) * bec.dx * bec.dy
  
  Ekin = sum( sum( bec.psi.conjugate() * f.fftshift( f.ifft2( -1 * bec.ksquare *
                        f.fft2( f.fftshift( bec.psi) ) ) ) ) ) * bec.dx * bec.dy
  
  norm      = sum(sum( bec.psi * ( bec.psi ).conjugate() )) * bec.dx * bec.dy
  
  print 'Potential Energy = ', Epot
  print 'Kinetic Energy   = ', Ekin
  print 'Total Energy     = ', Epot + Ekin
  print 'normalisation    = ', norm
  return Epot, Ekin

def psi_gravity_step(bec):
  '''
  GRAVITATIONALLY BOUND, SELF-INTERACTING BLOB!
  advance the wavefunction one timestep when the
  previous wavefunction is supplied as well as the
  exp(-0.5*k^2) vector, gravitational interaction
  strength GS, self-interaction strength gs and
  timestep DT
  '''

  
  #-----------------------------------------------------------
  #4th order 1-d integrator, kept in case a 4th order 2-d routine is needed
  #-----------------------------------------------------------
  
  def order2(c):
    Vc = np.exp( -1j * c * bec.dt / 2. * 
                   (1. * bec.V - bec.gravity() + bec.g * abs( bec.psi ) ** 2 ) )
    Tc = bec.expksquare ** c
    return Vc, Tc
  
  p = 1/(4.-4.**(1/3.))
  q = 1 - 4 * p
  
  Vp,Tp = order2(p)
  Vq,Tq = order2(q)
  
  return Vp * f.fftshift( f.ifft2( Tp * f.fft2( f.fftshift( Vp ** 2 * 
              f.fftshift( f.ifft2( Tp * f.fft2( f.fftshift( Vp * Vq *
              f.fftshift( f.ifft2( Tq * f.fft2( f.fftshift( Vq * Vp * 
              f.fftshift( f.ifft2( Tp * f.fft2( f.fftshift( Vp ** 2 *  
              f.fftshift( f.ifft2( Tp * f.fft2( Vp * f.fftshift( bec.psi ) ) ) )
              ) ) ) )
              ) ) ) )
              ) ) ) )
              ) ) ) )
  #-------------------------------------------------------------
  

def twoblobs(x,y,npt, **kwargs):
  X,Y = np.meshgrid(x,y)
  xo1 = -2.
  xo2 = +2.
  sig1 = 1.
  pi = np.pi
  sig2 = 1.
  psi1 = 1.0/np.sqrt(2*pi*sig1**2) * np.exp(-((X-xo1)**2 + Y**2.) / (2*sig1**2)) 
  psi2 = 1.0/np.sqrt(2*pi*sig2**2) * np.exp(-((X-xo2)**2 + Y**2.) / (2*sig2**2))
  psi = psi1 + psi2
  norm = sum(sum(psi*psi.conjugate())) * abs(x[1]-x[0]) * abs(y[1]-y[0])
  psi = psi / np.sqrt(norm)
  return psi
  
def threeblobs(x,y,npt, **kwargs):
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
  
def gauss( x              ,
           y              ,
           npt            ,
           means = [0.,0.],
           sig   = [1.,1.],
           corr  = 0.     ):
  '''
  Define an initial wavefunction (Gaussian)
  '''
  if abs( corr ) >= 1:
    print '<twod.gauss> corr must be strictly between -1 and 1'
    
  wave = np.zeros([npt,npt], dtype = 'complex' )
  
  #Gaussian Parameters and preliminary calculations for speed
  xsig = sig[0]
  ysig = sig[1]
  xycorr = np.array( [ [ xsig**2.,           corr * xsig * ysig ],
                      [ corr * xsig * ysig, ysig**2.           ] ] )
  
  corrdet = det(xycorr)
  
  xycorrinv = 1. / corrdet * np.array( [ [  xycorr[1,1], -xycorr[0,1]  ],
                                        [ -xycorr[1,0],  xycorr[0,0]  ] ] )
              
  corrdetroot = np.sqrt(corrdet)
  def phase( X, Y , X0, Y0):
    return 1j*np.arctan2( ( Y - Y0) , ( X - X0) )
  
  print '<twod.gauss> combobulating initial wavefunction matrix'
  for jj in range( 0, npt ):
    for kk in range( 0, npt ):
      pos = np.array( [ x[jj], y[kk] ] ) - means
      wave[jj,kk] = 1. / (2. * np.pi * corrdetroot ) * (
         np.exp( -0.5 * np.dot( np.dot( pos.transpose(), xycorrinv ), pos ) ) )
  print '<twod.gauss> completed'
  return wave

  
def vortexgauss( x                 ,  #xvals
                 y                 ,  #yvals
                 npt               ,  #no. gridpoints
                 vort  = [0, 0, 0] ,  #X by 2. array of vortex locations
                 means = [0.,0.]   ,  #[x,y] centre of gaussian
                 sig   = [1.,1.]   ,  #[stddev_x, stddev_y] for gaussian
                 corr  = 0.        ): #xy correlation
  '''
  Define an initial wavefunction (Gaussian) with vortices!
  SYNTAX:
  def vortexgauss( x              ,  #xvals
                   y              ,  #yvals
                   npt            ,  #no. gridpoints
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
    
  wave = np.zeros([npt,npt], dtype = 'complex' )
  
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
  theta = np.zeros( [npt,npt], dtype = complex )
  for X0, Y0, ax in vort: theta += ax * 1j * np.arctan2( ( Y - Y0) , ( X - X0) )
  
  plt.figure("Initial Phase")
  plt.contourf(x,y,np.angle(np.exp(theta)),20)
  plt.xlabel(r'x ($a_0$)')
  plt.ylabel(r'y ($a_0$)')
  plt.show()
  
  
  #Now do the whole wavefunction
  for jj in range( 0, npt ):
    for kk in range( 0, npt ):
      pos = np.array( [ x[jj], y[kk] ] ) - means
      wave[jj,kk] = 1. / (2. * np.pi * corrdetroot ) * (
         np.exp( -0.5 * np.dot( np.dot( pos.transpose(), xycorrinv ), pos ) 
                ) * np.exp( theta[jj,kk] ) )
  print '<twod.vortexgauss> completed'
  
  return wave

  
def harm_trap( x, y, npt, P ):
  v = np.zeros([npt,npt], dtype = 'complex' )
  for ii in range(0,npt):
    for jj in range(0,npt):
      v[ii,jj] = P * 0.5 * (x[ii]**2.+y[jj]**2.)
  return v
  
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
  
class Bose:
  '''
  Establish the Bose class
  The Bose class defines an item that contains most relevant arrays
  that are passed to functions. Many of these are to optimise the routine
  '''
  def __init__( self, a, b, npt, init, g, G, rot, P, dt, **kwargs ):
    self.a   = a
    self.b   = b
    self.npt = npt
    self.P   = P
    self.g   = g
    self.G   = G
    self.rot = rot
    self.dt  = dt
    
    self.x  = np.linspace( a, b, npt )
    self.dx = abs( self.x[1] - self.x[0] )
    self.y  = self.x
    self.dy = abs( self.y[1] - self.y[0] )
    
    self.psi              = init      ( self.x, self.y, self.npt, **kwargs )
    self.ksquare , self.k = k2dimen   ( self.a, self.b, self.npt )
    self.V                = harm_trap ( self.x, self.y, self.npt, self.P )
    self.ker              = kernel    ( self.x, self.y, self.npt )
    
    self.expksquare       = np.exp    (-0.5j * self.dt * self.ksquare)
    
  #-----------------------------------------------------------------------------
  def gravity(self):
    '''
    Evaluate the gravitational field, with a call to Bose.gravity()
    Gravitaional field is the convolution of the density and the kernel
    Uses self.ker = 1/|r|, self.psi = wavefunction
    '''
    den = abs(self.psi)**2.  #calculate the probability density
    
    #return the convolution, after multiplying by scaled gravity and 
    #correcting for grid scaling (due to 2 forward FFTs, and only one inverse
    return self.G * self.dx * self.dy * f.fftshift( f.ifft2( f.fft2( 
           f.fftshift( den ) ) * abs( f.fft2( f.fftshift( self.ker ) ) ) ) )
  
  #-----------------------------------------------------------------------------
  def angular(self):
    '''
    return the angular momentum operator value at each point in the grid
    L = -i ( x * d/dy - y * d/dx )
    We assume a square grid centred on  (x,y) = (0,0)
    '''
    kx = k_vector( self.npt, self.b - self.a )
    ky = kx
    
    KX, KY = np.meshgrid( kx, ky )
    X ,  Y = np.meshgrid( self.x, self.y )    
    fpsi   = f.fft2( f.fftshift( self.psi ) )
    
    
    return self.rot *1j * ( X * f.fftshift( f.ifft2( -1j * KY * fpsi) ) -
                            Y * f.fftshift( f.ifft2( -1j * KX * fpsi) )
                          ) 
  #-----------------------------------------------------------------------------
  def step2(self):
    '''
    Perform a second order timestep
    '''
    
    V2 = np.exp( -1j * self.dt / 2. * (self.V - self.gravity() + 
                                       self.g * abs( self.psi ) ** 2 +
                                       self.angular()
                                       ) 
                )
                   
    return V2 * f.fftshift( f.ifft2(self.expksquare * 
                                    f.fft2( f.fftshift( V2 * self.psi ) )
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
                      self.g * abs( self.psi ) ** 2 +
                      self.angular()
                    )
                  )
      Tc = self.expksquare ** c
      return Vc, Tc
    
    p = 1/(4.-4.**(1/3.))
    q = 1 - 4 * p
    
    Vp,Tp = order2(p)
    Vq,Tq = order2(q)
    
    return Vp * f.fftshift( f.ifft2( Tp * f.fft2( f.fftshift( Vp ** 2 * 
                f.fftshift( f.ifft2( Tp * f.fft2( f.fftshift( Vp * Vq *
                f.fftshift( f.ifft2( Tq * f.fft2( f.fftshift( Vq * Vp * 
                f.fftshift( f.ifft2( Tp * f.fft2( f.fftshift( Vp ** 2 *  
                f.fftshift( f.ifft2( Tp * f.fft2( f.fftshift( Vp * self.psi 
                ) ) ) )
                ) ) ) )
                ) ) ) )
                ) ) ) )
                ) ) ) )
    
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

  
  ''' 
  Old code for FFTW routine that I couldn't get to work properly
  #Define and plan fourier transforms. Need to copy data into the timed
  #or freqd arrays to perform Fourier transforms
  #Also, you cannot change the memory pointers to timed and freqd.
  timed = np.zeros( [ npt, npt ] , dtype = 'complex' )
  freqd = np.zeros( [ npt, npt ] , dtype = 'complex' )
  
  fft  = fftw.Plan( timed, freqd, direction = 'forward'  ) #forward FFT
  ifft = fftw.Plan( freqd, timed, direction = 'backward' ) #inverse FFT
  '''
  
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
              erase    = False       ,  #write over existing files True/False
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
                  erase    = False       ,  #write over existing files True/False
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
  print 'Run saved to ' + filename
  print 'Parameters added to sims/runs.info'
  print 'Have a nice day :)'
  infile.f.close()

def nicecmap():
  '''
  Nice colormap that looks good in greyscale
  From: Rappaport 2002, doi:10.1109/MAP.2002.1028735
  '''
  cdict = {'red'   : ((0.000, 0.00, 0.00),
                    (0.125, 0.15, 0.15),
                    (0.250, 0.30, 0.30),
                    (0.375, 0.60, 0.60),
                    (0.500, 1.00, 1.00),
                    (0.625, 0.90, 0.90),
                    (0.750, 0.90, 0.90),
                    (0.875, 0.90, 0.90),
                    (1.000, 1.00, 1.00)),
         
         'green' : ((0.000, 0.00, 0.00),
                    (0.125, 0.15, 0.15),
                    (0.250, 0.15, 0.15),
                    (0.375, 0.20, 0.20),
                    (0.500, 0.25, 0.25),
                    (0.625, 0.50, 0.50),
                    (0.750, 0.75, 0.75),
                    (0.875, 0.90, 0.90),
                    (1.000, 1.00, 1.00)),
         
         'blue'  : ((0.000, 0.00, 0.00),
                    (0.125, 0.50, 0.50),
                    (0.250, 0.75, 0.75),
                    (0.375, 0.50, 0.50),
                    (0.500, 0.15, 0.15),
                    (0.625, 0.00, 0.00),
                    (0.750, 0.10, 0.10),
                    (0.875, 0.50, 0.50),
                    (1.000, 1.00, 1.00))
         }
  return matplotlib.colors.LinearSegmentedColormap('rgb-grey_cmap',cdict,256)
  
def twodani(filename):
  '''
  Plot the simulation generated in the file filename.
  Filename is a binary hd5 file generated by twodsave.
  '''
  
  #Setup------------------------------------------------------------------------
  infile = h5file( filename, erase = False, read = True )
  fig = plt.figure()
  
  steps = infile.head[ 'steps' ]
  npt   = infile.head[ 'npt' ]
  
  xvals, yvals = infile.readxy() #read x-y array
  
  p = infile.readpsi('0.0') #read initial state
  
  #Plot probability density-----------------------------------------------------
  
  ax = fig.add_subplot(221, aspect = 'equal')
  
  plt.xlim( [ -5, 5 ] )
  plt.ylim( [ -5, 5 ] )
  
  plt.xlabel(r'x ($a_0$)')
  plt.ylabel(r'y ($a_0$)')
  
  im = plt.imshow( abs( p ) ** 2 / ( abs( p ) ** 2 ).max(),
                  extent = ( min(xvals), max(xvals), min(yvals), max(yvals) ),
                  vmin = 0., vmax = 1., cmap = nicecmap() ) #(abs(p)**2).max())
 
  cbar = plt.colorbar(im, ticks = np.linspace( 0, 1, 5 ) )
  
  #if not autoscaling density:
  #v = np.linspace(0, ( abs( p ) ** 2. ).max() , 10)
  #for jj in range(0,len(v)): v[jj] = round(v[jj],3)
  #cbar = plt.colorbar(ticks = v)
  
  #Phase------------------------------------------------------------------------
  
  ax2 = fig.add_subplot( 222, aspect = 'equal' )
  plt.xlim( [ -5, 5 ] )
  plt.ylim( [ -5, 5 ] )
 
  plt.xlabel(r'x ($a_0$)')
  plt.ylabel(r'y ($a_0$)')
  
  pi = np.pi
  im2 = plt.imshow( np.angle( p ),
                    extent = ( min(xvals), max(xvals), min(yvals), max(yvals) ),
                    vmin = -np.pi, vmax = np.pi, cmap=cm.gist_rainbow ) 
  cbar2 = plt.colorbar(im2,
                       ticks = [round(j,2) for j in np.linspace(-pi,pi,5)]
                      )
  
  #--------------------------------------------------------
  
  #x = 0.
  ax3 = fig.add_subplot( 223 )
  plt.xlim( [-10,10] )
  plt.ylim( [0,1] )
 
  plt.xlabel(r'x ($a_0$)')
  plt.ylabel(r'$|\psi|^2$ (atoms $a_{0}^{-1}$)')
  
  line1, = ax3.plot( xvals, ( abs( p ) ** 2. )[ npt / 2., : ] / 
                          ( abs( p ) ** 2. )[ npt / 2., : ].max() )
    
  #--------------------------------------------------------
  
  #y = 0.
  
  ax4 = fig.add_subplot(224)
  plt.xlim( [-10,10] )
  plt.ylim( [0, 1] )
  
  plt.xlabel( r'y ($a_0$)' )
  plt.ylabel( r'$|\psi|^2$ (atoms $a_{0}^{-1}$)' )
  
  line2, = ax4.plot( yvals, ( abs( p ) **2. )[ : , npt / 2. ] / 
                          ( abs( p ) **2. )[ : , npt / 2.].max() )
  
  #Update the plot--------------------------------------------------------------
  def update_fig(i):
    p = infile.readpsi( str(i) )
    
    im.set_array( abs( p ) ** 2 / (abs(p)**2).max() )
    im2.set_array( np.angle( p ) )
    line1.set_ydata( ( abs( p ) ** 2. )[ npt / 2., : ] / 
                     ( abs( p ) ** 2. )[ npt / 2., : ].max() )
    line2.set_ydata( ( abs( p ) ** 2. )[ :, npt / 2. ] / 
                     ( abs( p ) ** 2. )[ :, npt / 2. ].max() )
    
    return im, im2, line1, line2

  ani = animation.FuncAnimation( fig, update_fig,
                                 np.arange( 1, steps + 1., dtype=float ),
                                 interval=20., repeat_delay = 3000., 
                                 blit = False )
  plt.show()
  return ani
  infile.f.close()
  

def aniprob(filename):
  '''
  Plot the simulation generated in the file filename, plotting only 
  the probability density
  Filename is a binary hd5 file generated by twodsave.
  '''
  
  #Setup------------------------------------------------------------------------
  infile = h5file( filename, erase = False, read = True )
  fig = plt.figure()
  matplotlib.rcParams.update({'font.size':16}) #set inital font size
  
  steps = infile.head[ 'steps' ]
  npt   = infile.head[ 'npt' ]
  
  xvals, yvals = infile.readxy() #read x-y array
  
  p = infile.readpsi('0.0') #read initial state
  
  #Plot probability density-----------------------------------------------------
  
  ax = fig.add_subplot(111, aspect = 'equal')
  
  #plt.xlim( [ -5, 5 ] )
  #plt.ylim( [ -5, 5 ] )
  
  plt.xlim( [ -10, 10 ] )
  plt.ylim( [ -10, 10 ] )
  
  plt.xlabel(r'x ($a_0$)')
  plt.ylabel(r'y ($a_0$)')
  
  im = plt.imshow( abs( p ) ** 2 / ( abs( p ) ** 2 ).max(),
                  extent = ( min(xvals), max(xvals), min(yvals), max(yvals) ),
                  vmin = 0., vmax = 1., cmap = nicecmap() ) #(abs(p)**2).max())
 
  cbar = plt.colorbar(im, ticks = np.linspace( 0, 1, 5 ) )
  
  #if not autoscaling density:
  #v = np.linspace(0, ( abs( p ) ** 2. ).max() , 10)
  #for jj in range(0,len(v)): v[jj] = round(v[jj],3)
  #cbar = plt.colorbar(ticks = v)
  
  
  #Update the plot--------------------------------------------------------------
  def update_fig(i):
    p = infile.readpsi( str(i) )
    
    im.set_array( abs( p ) ** 2 / (abs(p)**2).max() )
        
    return im,

  ani = animation.FuncAnimation( fig, update_fig,
                                 np.arange( 1, steps + 1., dtype=float ),
                                 interval=20., repeat_delay = 3000., 
                                 blit = False )
  plt.show()
  return ani
  infile.f.close()

def aniphase(filename):
  '''
  Plot the simulation generated in the file filename, plotting only 
  the phase
  Filename is a binary hd5 file generated by twodsave.
  '''
  
  #Setup------------------------------------------------------------------------
  infile = h5file( filename, erase = False, read = True )
  fig = plt.figure()
  matplotlib.rcParams.update({'font.size':16}) #set inital font size
  
  steps = infile.head[ 'steps' ]
  npt   = infile.head[ 'npt' ]
  
  xvals, yvals = infile.readxy() #read x-y array
  
  p = infile.readpsi('0.0') #read initial state
  
  #Plot phase-------------------------------------------------------------------
  
  ax = fig.add_subplot(111, aspect = 'equal')
  
  #plt.xlim( [ -5, 5 ] )
  #plt.ylim( [ -5, 5 ] )
  
  plt.xlim( [ -10, 10 ] )
  plt.ylim( [ -10, 10 ] )
  
  plt.xlabel(r'x ($a_0$)')
  plt.ylabel(r'y ($a_0$)')
  
  pi = np.pi
  im = plt.imshow( np.angle( p ),
                    extent = ( min(xvals), max(xvals), min(yvals), max(yvals) ),
                    vmin = -np.pi, vmax = np.pi, cmap=cm.gist_rainbow ) 
  cbar = plt.colorbar(im,
                       ticks = [round(j,2) for j in np.linspace(-pi,pi,5)]
                      )
  
  
  #Update the plot--------------------------------------------------------------
  def update_fig(i):
    p = infile.readpsi( str(i) )
    
    im.set_array( np.angle( p ) )
        
    return im,

  ani = animation.FuncAnimation( fig, update_fig,
                                 np.arange( 1, steps + 1., dtype=float ),
                                 interval=20., repeat_delay = 3000., 
                                 blit = False )
  plt.show()
  return ani
  infile.f.close() 

def fileinfo(filename):
  '''
  Print out header data for a HDF5 file
  '''
  print "Header data for " + str(filename)
  try: infile = h5file( filename, erase = False, read = True )
  except: return SystemError(str(filename) + " is not a valid hdf5 file")
  infile.f.close()
  print str(filename) + ' closed'
  
def evenvortex(rad,num):
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
  