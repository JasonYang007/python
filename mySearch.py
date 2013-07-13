import re, os,sys, types
from stat import *

def mySearch(pat, path, recursive=False, verbose=False):
	cpat=re.compile(pat) if type(pat) is types.StringType else pat
	try:
			fList=os.listdir(path)
	except : 
			return	
	for ff in os.listdir(path):
		apath=os.path.join(path, ff)
		try:
			if recursive and S_ISDIR(os.stat(apath)[ST_MODE]):
				mySearch(pat,apath,recursive)
			else:
				if verbose: print 'Checking %s' % apath
				with open(apath,'r') as ifd:
					for idx,ll in enumerate(ifd.readlines()):
						if cpat.search(ll):
							print '%s at line %d' % (apath, idx+1)
		except IOError as (errno, strerror):
			print 'I/O error(%d): %s for %s' % (errno, strerror, apath)
		except:
			print 'Unexpected error:', sys.exc_info()[0]

if __name__ == '__main__':
	from optparse import OptionParser
	usage='%pgm [-r] pattern dir'
	parser=OptionParser(usage)
	parser.add_option('-r','--recursive',
		action='store_true', dest='recursive', default=False)
	parser.add_option('-v','--verbose',
		action='store_true', dest='verbose', default=False)
	(options, args)=parser.parse_args()
	mySearch(args[0], args[1], options.recursive, options.verbose)


		

