import numpy as np
import copy as cp

def shear(ice,alpha,crop = True):
	""" Applies a shear transformation to an ice object. The crop option keeps the square geometry of the overall system"""
	def shear_transform(array, alpha):
		m = np.cos(alpha)
		return np.array([
			array[:,0]+m*array[:,1],
			array[:,1]+(np.sqrt(1-m**2)-1)*array[:,1],
			array[:,2]]).transpose()

	def crop_sheared(centers,width):
		centers[:,0] = np.mod(centers[:,0],width)
		return centers

	units = ice[0].center.units
	centers = np.array([s.center.to(units).magnitude for s in ice])
	directions = np.array([s.direction.to(units).magnitude for s in ice])

	centers_sheared = shear_transform(centers,alpha)
	directions_sheared = shear_transform(directions,alpha)

	if crop:
		centers_sheared = crop_sheared(centers_sheared,max(centers[:,0]))
	
	ice_sheared = cp.deepcopy(ice)
	for i,s in enumerate(ice_sheared):
		ice_sheared[i].center = centers_sheared[i,:]*units
		ice_sheared[i].direction = directions_sheared[i,:]*units
	
	return ice_sheared
	
def rotate(ice,theta):
	
	rot = lambda angle: np.array([
		[np.cos(angle),-np.sin(angle),0],
		[np.sin(angle),np.cos(angle),0],
		[0,0,1]])
		
	ice_rotated = cp.deepcopy(ice)
	for i,s in enumerate(ice):
		ice_rotated[i].center = np.matmul(rot(theta),ice[i].center)*ice[i].center.units
		ice_rotated[i].direction = np.matmul(rot(theta),ice[i].direction)*ice[i].direction.units
		
	return ice_rotated
	
def scale(ice,factor):

	try:
		if len(factor)==2:
			try:
				factor = np.append(factor,1)*factor.units
			except AttributeError:
				factor = np.append(factor,1)
	except TypeError:
		factor = np.array([1,1,1])*factor
	
	ice_scaled = cp.deepcopy(ice)
	for i,s in enumerate(ice):
		
		ice_scaled[i].center = ice[i].center*factor
		ice_scaled[i].direction = ice[i].direction*factor
		
	return ice_scaled
	
def translate(ice,vector):
	
	if len(vector)==2:
		try:
			vector = np.append(vector,0)*vector.units
		except AttributeError:
			vector = np.append(vector,0)
				
	ice_translated = cp.deepcopy(ice)
	for i,s in enumerate(ice):
		ice_translated[i].center = ice[i].center+vector
		
	return ice_translated