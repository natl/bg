#######################################################################
#
# Program:      HDF 5 File Handling for trap simulations
#
# Author:       Nathanael Lampe, School of Physics, Monash University
#
# Created:      December 4, 2012
#
# Changelog:    December 4, 2012: Created from twodtrap.py
#           
#               
#
# Purpose:      Contain functions that do most of the hdf5 file handling used
#               in routines
#               
########################################################################

#Imports
from __future__ import division

import h5py

def autof(h              ,
          folder = 'sims'):
  '''
  Automatically get file database, make a filename, and add headers to a 
  database file
  SYNTAX autof(h              ,
               folder = 'sims'): #option to change the directory for
                                 #different simulation types
  
  
  headers are specified by h
  '''
  dfile = open(folder + '/runs.info', 'r+')
  
  run = len( dfile.readlines() ) - 2. #How many lines in the file?
                                      #run = file lines - 1 (due to
                                      #file structure.
  
  filename = (folder + "/tdgb%i.hdf5" % run) #Two-D Gravitational Bec X .hdf5
  
  v=[]
  for row in h[ 'vortices' ]: v.append( list( row ) ) #get vortices in 
                                                      #a nice format
  
  dfile.write( '%4i      %5s     %8f  %8f  %8f  %8f  %8f  %8f  %8f  %8f  %8f  %8f  %8f  %s\n' %
               ( run, h['wick'], h['G'], h['g'], h['rot'], h['P'], h['dt'], h['tstop'],
                 h['xmin'], h['xmax'], h['npt'], h['skipstep'], h['steps'],
                 str(v) )
              )
  
  
  return filename
  
########################################################################

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
  
########################################################################

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
  
########################################################################
  
def fileinfo(filename):
  '''
  Print out header data for a HDF5 file
  '''
  print "Header data for " + str(filename)
  try: infile = h5file( filename, erase = False, read = True )
  except: return SystemError(str(filename) + " is not a valid hdf5 file")
  infile.f.close()
  print str(filename) + ' closed'
  
########################################################################
  
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

