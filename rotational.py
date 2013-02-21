################################################################################
#
# Program:      2D BEC simulation runs - angular momentum runs
#
# Author:       Nathanael Lampe, School of Physics, Monash University
#
# Created:      January 23, 2012
#
# Changelog:    twodtrap.py
#               Aug 29, 2012: Created
#               Oct 11, 2012: Working, changelog begun
#               Oct 11, 2012: Implementation of rotation into Hamiltonian
#               twod.py
#               Dec 04, 2012: Created from twodtrap.py
#               Dec 12, 2012: Implemented scheme to check cnvergence against the
#                             chemical potential
#               Jan 24, 2013: The program works in accordance with the virial 
#                             theorem, and TF approximation, Has now been forked
#                             to rotate.py, where rotation will be implemented.
#               irrotational.py
#               Jan 30, 2013: Forked from the twod.py runtime, to execute
#                             specific scenarios
#           --->rotational.py
#               Feb 12, 2013: Forked from the irrotational.py program to execute
#                             required scenarios
#
# Purpose:      Provide a function to evaluate the stability of a 
#               gravitationally bound BEC under varying trap strengths
#               
################################################################################

#Imports:
from __future__ import division

import numpy as np

from scripts.initials import *
from scripts.trapsave import *
from scripts.trapplot import *
from scripts.rotate   import *
################################################################################

def rotatebec( a                      ,  #min co-ord
               b                      ,  #max co-ord
               npt                    ,  #no. gridpoints
               dt                     ,  #timestep
               tstop                  ,  #time to stop
               g                      ,  #s-wave scattering strength
               G                      ,  #gravitational field scaled strength
               wick     = False       ,  #Wick rotation T/F?
               rot      = 0.          ,  #angular momentum in system
               filename = autof       ,  #output filename, autof for automatic naming
               P        = 0.          ,  #optional harmonic potential
               init     = gravground  ,  #initial wavefunction shape (function call)
               skip     = 1.          ,  #intervals to skip when saving
               erase    = False       ,  #overwrite existing files True/False
               order    = 5           ,  #Rotational FEM order
               **kwargs                ):
  '''
  save the result of a gravitational simulation of a BEC
  saved file is a hd5 database
  SYNTAX:
  def rotatebec( a                      ,  #min co-ord
                 b                      ,  #max co-ord
                 npt                    ,  #no. gridpoints
                 dt                     ,  #timestep
                 tstop                  ,  #time to stop
                 g                      ,  #s-wave scattering strength
                 G                      ,  #gravitational field scaled strength
                 wick     = False       ,  #Wick rotation T/F?
                 rot      = 0.          ,  #angular momentum in system
                 filename = autof       ,  #output filename, autof for automatic naming
                 P        = 0.          ,  #optional harmonic potential
                 init     = gravground  ,  #initial wavefunction shape (function call)
                 skip     = 1.          ,  #intervals to skip when saving
                 erase    = False       ,  #overwrite existing files True/False
                 order    = 5           ,  #Rotational FEM order
                 **kwargs                ):
  '''
  #initial setup ---------------------------------------------------------------
  assert type(wick) == bool, 'wick is a boolean, ie. True or False'
  
  #Prepare parameters for database
  h = dict( {'G'        : G                    ,
             'g'        : g                    ,
             'rot'      : rot                  ,
             'P'        : P                    ,
             'dt'       : dt                   ,
             'tstop'    : tstop                ,
             'xmin'     : a                    ,
             'xmax'     : b                    ,
             'npt'      : npt                  ,
             'skipstep' : skip                 ,
             'steps'    : (tstop // dt) // skip,
             'wick'     : wick                 ,
             'order'    : order                ,} )
  h = dict( h.items() + kwargs.items() ) #merge the dictionaries
  
  if filename == autof: #automatically add run to database, and name file
    filename = autof(h, folder = 'rotate')
  else: #custom filename
    if type(filename) != str: return SystemError('Filename should be a string')
  
  #Make a condensate
  bec = Bose( a, b, int(npt), init, g, G, rot, P, dt, order = order, **kwargs)  
  
  if wick == True:
    bec.wickon()
  else:
    bec.wickoff()
  
  infile = h5file( filename, erase = erase, read = False )
  infile.add_headers( h )
  infile.add_data( str('0.0'), bec.x, bec.y, bec.psi, time=0.0) 
  
  bec.energies( verbose = True )
  savecounter = 0.
  saves=0
  # time-evolution------------------------------------------------------------
  for t in np.arange( 0, tstop, dt ):
    bec.stepRK4() #timestep bec.psi, Wick rotation is handled in stepRK4
    savecounter += 1.
    
    if savecounter == skip: #save the data
      infile.add_data(str( saves + 1.), bec.x, bec.y, bec.psi, t)
      savecounter = 0.
      saves += 1
      print t
  
  bec.energies( verbose = True )
  
  print 'Run saved to ' + filename
  print 'Parameters added to sims/runs.info'
  print 'Have a nice day :)'
  infile.f.close()
  
  return None


def explode( a                      ,  #min co-ord
             b                      ,  #max co-ord
             npt                    ,  #no. gridpoints
             dt                     ,  #timestep
             tstop                  ,  #time to stop
             g                      ,  #s-wave scattering strength
             G                      ,  #gravitational field scaled strength
             mult                   ,  #multiplier for g - explosion/collapse
             rot      = 0.          ,  #angular momentum in system
             filename = autof       ,  #output filename, autof for automatic naming
             P        = 0.          ,  #optional harmonic potential
             init     = gauss       ,  #initial wavefunction shape (function call)
             skip     = 1.          ,  #intervals to skip when saving
             erase    = False       ,  #overwrite existing files True/False
             tol      = 0.1         ,  #convergence tolerance
             **kwargs                ):
  '''
  save the result of a gravitational simulation of a BEC
  saved file is a hd5 database
  SYNTAX:
  def explode( a                      ,  #min co-ord
               b                      ,  #max co-ord
               npt                    ,  #no. gridpoints
               dt                     ,  #timestep
               tstop                  ,  #time to stop
               g                      ,  #s-wave scattering strength
               G                      ,  #gravitational field scaled strength
               mult                   ,  #multiplier for g - explosion/collapse
               rot      = 0.          ,  #angular momentum in system
               filename = autof       ,  #output filename, autof for automatic naming
               P        = 0.          ,  #optional harmonic potential
               init     = gauss       ,  #initial wavefunction shape (function call)
               skip     = 1.          ,  #intervals to skip when saving
               erase    = False       ,  #overwrite existing files True/False
               tol      = 0.1         ,  #convergence tolerance
               **kwargs                ):
  '''
  #initial setup ---------------------------------------------------------------
  
  #Prepare parameters for database
  h = dict( {'G'        : G                    ,
             'g'        : g                    ,
             'rot'      : rot                  ,
             'P'        : P                    ,
             'dt'       : dt                   ,
             'tstop'    : tstop                ,
             'xmin'     : a                    ,
             'xmax'     : b                    ,
             'npt'      : npt                  ,
             'skipstep' : skip                 ,
             'steps'    : (tstop // dt) // skip,
             'mult'     : mult                 ,
             'tol'      : tol                  } )
  h = dict( h.items() + kwargs.items() ) #merge the dictionaries
  
  if filename == autof: #automatically add run to database, and name file
    filename = autof(h, folder = 'collapse')
  else: #custom filename
    if type(filename) != str: return SystemError('Filename should be a string')
  
  #Make a condensate
  bec = Bose( a, b, int(npt), init, g, G, rot, P, dt, **kwargs)  
  t = 0
  
  infile = h5file( filename, erase = erase, read = False )
  infile.add_headers( h )
  infile.add_data( str('0.0'), bec.x, bec.y, bec.psi, t )
  

################################################################################
# Find the ground state
################################################################################
  bec.wickon()
  savecounter = 0.
  saves       = 0.
  converge    = False
  oldvirial   = bec.virial( verbose = False )
  olddiff     = -1 #set the olddiff to a negative number temporarily
  
  #Iterate until we have found the ground state
  while converge == False:
    bec.psi = bec.step4()
    savecounter += 1.
    
    norm  = sum( sum( abs( bec.psi )**2 ) ) * bec.dx * bec.dy
    bec.psi = bec.psi / np.sqrt(norm)
      
    if savecounter == skip: #save the data
      saves += 1.
      infile.add_data(str( saves ), bec.x, bec.y, bec.psi, t)
      savecounter = 0.
      virial = bec.virial( verbose = True )
      diff = abs(oldvirial - virial)
      if virial <= tol:
        converge = True
        infile.add_headers( { 'Convergene' : 'Success!' } )
      
      elif diff == 0.:
        print '''The initial conditions are insufficient for convergence to the 
                 desired tolerance'''
        print '''Terminating Run'''
        infile.add_headers( { 'Convergene' : 'Failed' } )
        break
      
      olddiff = diff
      print t, converge
    t += dt
  
    t0 = t
################################################################################
# explode it if the ground state is found, else end
################################################################################
  if converge == True:
    savecounter = 0.
    
    bec.wickoff()
    bec.g *= mult #multiply g by the multiplier specified
    
    # time-evolution------------------------------------------------------------
    for t in np.arange( t0, tstop + t0, dt ):
      bec.psi = bec.step4()
      savecounter += 1.
      
      if savecounter == skip: #save the data
        infile.add_data(str( saves + 1.), bec.x, bec.y, bec.psi, t)
        savecounter = 0.
        saves += 1.
        print t
  
  print 'Run saved to ' + filename
  print 'Parameters added to sims/runs.info'
  print 'Have a nice day :)'
  infile.f.close()
  
  return None

def threebec( a                      ,  #min co-ord
              b                      ,  #max co-ord
              npt                    ,  #no. gridpoints
              dt                     ,  #timestep
              tstop                  ,  #time to stop
              g                      ,  #s-wave scattering strength
              G                      ,  #gravitational field scaled strength
              sep                    ,  #separation of bec's from center
              rot      = 0.          ,  #angular momentum in system
              filename = autof       ,  #output filename, autof for automatic naming
              P        = 0.          ,  #optional harmonic potential
              init     = gravground  ,  #initial wavefunction shape (function call)
              skip     = 1.          ,  #intervals to skip when saving
              erase    = False       ,  #overwrite existing files True/False
              tol      = 0.1         ,  #convergence tolerance
              **kwargs                ):
  '''
  save the result of a gravitational simulation of a BEC
  saved file is a hd5 database
  SYNTAX:
  def threebec( a                      ,  #min co-ord
                b                      ,  #max co-ord
                npt                    ,  #no. gridpoints
                dt                     ,  #timestep
                tstop                  ,  #time to stop
                g                      ,  #s-wave scattering strength
                G                      ,  #gravitational field scaled strength
                sep                    ,  #separation of bec's from center
                rot      = 0.          ,  #angular momentum in system
                filename = autof       ,  #output filename, autof for automatic naming
                P        = 0.          ,  #optional harmonic potential
                init     = gauss       ,  #initial wavefunction shape (function call)
                skip     = 1.          ,  #intervals to skip when saving
                erase    = False       ,  #overwrite existing files True/False
                tol      = 0.1         ,  #convergence tolerance
                **kwargs                ):
  '''
  #initial setup ---------------------------------------------------------------
  
  #Prepare parameters for database
  h = dict( {'G'        : G                    ,
             'g'        : g                    ,
             'rot'      : rot                  ,
             'P'        : P                    ,
             'dt'       : dt                   ,
             'tstop'    : tstop                ,
             'xmin'     : a                    ,
             'xmax'     : b                    ,
             'npt'      : npt                  ,
             'skipstep' : skip                 ,
             'steps'    : (tstop // dt) // skip,
             'sep'      : sep                 ,
             'tol'      : tol                  } )
  h = dict( h.items() + kwargs.items() ) #merge the dictionaries
  
  if filename == autof: #automatically add run to database, and name file
    filename = autof(h, folder = 'threebec')
  else: #custom filename
    if type(filename) != str: return SystemError('Filename should be a string')
  
  #Make a condensate
  bec = Bose( a, b, int(npt), init, g, G, rot, P, dt, **kwargs)  
  t = 0
  
  infile = h5file( filename, erase = erase, read = False )
  infile.add_headers( h )
  infile.add_data( str('0.0'), bec.x, bec.y, bec.psi, t )
  

################################################################################
# Find the ground state
################################################################################
  bec.wickon()
  savecounter = 0.
  saves       = 0.
  converge    = False
  oldvirial   = bec.virial( verbose = False )
  olddiff     = -1 #set the olddiff to a negative number temporarily
  
  #Iterate until we have found the ground state
  while converge == False:
    bec.psi = bec.step4()
    savecounter += 1.
    
    norm  = sum( sum( abs( bec.psi )**2 ) ) * bec.dx * bec.dy
    bec.psi = bec.psi / np.sqrt(norm)
      
    if savecounter == skip: #save the data
      saves += 1.
      infile.add_data(str( saves ), bec.x, bec.y, bec.psi, t)
      savecounter = 0.
      virial = bec.virial( verbose = True )
      diff = abs(oldvirial - virial)
      if virial <= tol:
        converge = True
        infile.add_headers( { 'Convergene' : 'Success!' } )
      
      elif diff == 0.:
        print '''The initial conditions are insufficient for convergence to the 
                 desired tolerance'''
        print '''Terminating Run'''
        infile.add_headers( { 'Convergene' : 'Failed' } )
        break
      
      olddiff = diff
      print t, converge
    t += dt
  t0 = t
################################################################################
# Rearrange the ground state into three blobs
################################################################################
  gs = bec.psi
  d  = int( sep / bec.dx )
  dv = int( d / 2 )
  dh = int( np.sqrt(3) * dv )
  #use roll to modify matrix, copy three times and then normalise
  bec.psi  = np.roll( gs, -d, axis = 1 )
  bec.psi += np.roll( np.roll(gs, dv, axis = 1),  dh, axis = 0 )
  bec.psi += np.roll( np.roll(gs, dv, axis = 1), -dh, axis = 0 )
  
  norm  = sum( sum( abs( bec.psi )**2 ) ) * bec.dx * bec.dy
  bec.psi = bec.psi / np.sqrt(norm)
  
################################################################################
# Then iterate until happy!
################################################################################
  if converge == True:
    savecounter = 0.
    
    bec.wickoff()
    
    # time-evolution------------------------------------------------------------
    for t in np.arange( t0, tstop + t0, dt ):
      bec.psi = bec.step4()
      savecounter += 1.
      
      if savecounter == skip: #save the data
        infile.add_data(str( saves + 1.), bec.x, bec.y, bec.psi, t)
        savecounter = 0.
        saves += 1.
        print t
  
  print 'Run saved to ' + filename
  print 'Parameters added to sims/runs.info'
  print 'Have a nice day :)'
  infile.f.close()
  
  return None  

def twobec( a                      ,  #min co-ord
            b                      ,  #max co-ord
            npt                    ,  #no. gridpoints
            dt                     ,  #timestep
            tstop                  ,  #time to stop
            g                      ,  #s-wave scattering strength
            G                      ,  #gravitational field scaled strength
            sep                    ,  #separation of bec's from center
            rot      = 0.          ,  #angular momentum in system
            filename = autof       ,  #output filename, autof for automatic naming
            P        = 0.          ,  #optional harmonic potential
            init     = gravground  ,  #initial wavefunction shape (function call)
            skip     = 1.          ,  #intervals to skip when saving
            erase    = False       ,  #overwrite existing files True/False
            tol      = 0.1         ,  #convergence tolerance
            k        = [0., 1.]    ,  #direction of initial 'kick', [kx,ky]
            **kwargs                ):
  '''
  save the result of a gravitational simulation of a BEC
  saved file is a hd5 database
  SYNTAX:
  def twobec( a                      ,  #min co-ord
              b                      ,  #max co-ord
              npt                    ,  #no. gridpoints
              dt                     ,  #timestep
              tstop                  ,  #time to stop
              g                      ,  #s-wave scattering strength
              G                      ,  #gravitational field scaled strength
              sep                    ,  #separation of bec's from center
              rot      = 0.          ,  #angular momentum in system
              filename = autof       ,  #output filename, autof for automatic naming
              P        = 0.          ,  #optional harmonic potential
              init     = gravground  ,  #initial wavefunction shape (function call)
              skip     = 1.          ,  #intervals to skip when saving
              erase    = False       ,  #overwrite existing files True/False
              tol      = 0.1         ,  #convergence tolerance
              k        = [0., 1.]    ,  #direction of initial 'kick', [kx,ky]
              **kwargs                ):
  '''
  #initial setup ---------------------------------------------------------------
  
  #Prepare parameters for database
  h = dict( {'G'        : G                    ,
             'g'        : g                    ,
             'rot'      : rot                  ,
             'P'        : P                    ,
             'dt'       : dt                   ,
             'tstop'    : tstop                ,
             'xmin'     : a                    ,
             'xmax'     : b                    ,
             'npt'      : npt                  ,
             'skipstep' : skip                 ,
             'steps'    : (tstop // dt) // skip,
             'sep'      : sep                  ,
             'tol'      : tol                  ,
             'k'        : k                    } )
  h = dict( h.items() + kwargs.items() ) #merge the dictionaries
  
  if filename == autof: #automatically add run to database, and name file
    filename = autof(h, folder = 'twobec')
  else: #custom filename
    if type(filename) != str: return SystemError('Filename should be a string')
  
  #Make a condensate
  bec = Bose( a, b, int(npt), init, g, G, rot, P, dt, **kwargs)  
  t = 0
  
  infile = h5file( filename, erase = erase, read = False )
  infile.add_headers( h )
  infile.add_data( str('0.0'), bec.x, bec.y, bec.psi, t )
  

################################################################################
# Find the ground state
################################################################################
  bec.wickon()
  savecounter = 0.
  saves       = 0.
  converge    = False
  oldvirial   = bec.virial( verbose = False )
  olddiff     = -1 #set the olddiff to a negative number temporarily
  
  #Iterate until we have found the ground state
  while converge == False:
    bec.psi = bec.step4()
    savecounter += 1.
    
    norm  = sum( sum( abs( bec.psi )**2 ) ) * bec.dx * bec.dy
    bec.psi = bec.psi / np.sqrt(norm)
      
    if savecounter == skip: #save the data
      saves += 1.
      infile.add_data(str( saves ), bec.x, bec.y, bec.psi, t)
      savecounter = 0.
      virial = bec.virial( verbose = True )
      diff = abs(oldvirial - virial)
      if virial <= tol:
        converge = True
        infile.add_headers( { 'Convergene' : 'Success!' } )
      
      elif diff == 0.:
        print '''The initial conditions are insufficient for convergence to the 
                 desired tolerance'''
        print '''Terminating Run'''
        infile.add_headers( { 'Convergene' : 'Failed' } )
        break
      
      olddiff = diff
      print t, converge
    t += dt
  t0 = t
################################################################################
# Rearrange the ground state into three blobs
################################################################################
  gs = bec.psi
  d  = int( sep / bec.dx )
  #use roll to modify matrix, copy twice and then normalise
  bec.psi  = np.roll( gs * np.exp( 1j * (k[0] * bec.X + k[1] * bec.Y ) ),
                      d, axis = 1 )
  bec.psi += np.roll( gs, -d, axis = 1 )
  
  norm  = sum( sum( abs( bec.psi )**2 ) ) * bec.dx * bec.dy
  bec.psi = bec.psi / np.sqrt(norm)
  
################################################################################
# Then iterate until happy!
################################################################################
  if converge == True:
    savecounter = 0.
    
    bec.wickoff()
    
    # time-evolution------------------------------------------------------------
    for t in np.arange( t0, tstop + t0, dt ):
      bec.psi = bec.step4()
      savecounter += 1.
      
      if savecounter == skip: #save the data
        infile.add_data(str( saves + 1.), bec.x, bec.y, bec.psi, t)
        savecounter = 0.
        saves += 1.
        print t
  
  print 'Run saved to ' + filename
  print 'Parameters added to sims/runs.info'
  print 'Have a nice day :)'
  infile.f.close()
  
  return None

