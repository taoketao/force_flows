# A vectorized adaptation of my-simulation.py that uses numpy
# vector operations for simulation speed efficiency.

'''
itinerary:
- Make debugging line drawings
- Change plot rendering to take LRUD <>^v to plot points.
  Convert the system so that all the math done on the actual domain
  is then rendered to the pane in one stroke.
'''


import numpy as np
import random
import pygame
import time

def exit(): import sys; _=input('exiting.'); sys.exit()

SCREENSIZE = (120*9, 120*3) # 120 is nice and divisible
epsilon_touching = 1e2


class Pane(object):
    def __init__(self, global_screen, **kwargs):
        self.pane_coords_fractions = kwargs.get('pane_coords_fractions', (0,0))
        self.pane_location = tuple([int(self.pane_coords_fractions[i]*\
                SCREENSIZE[i]) for i in [0,1]])
        self.pane_size_fractions = kwargs.get('pane_size_fractions', (1,1))
        self.global_screen = global_screen
        self.my_screensize = np.array([\
                int(global_screen.get_width() *self.pane_size_fractions[0]),
                int(global_screen.get_height()*self.pane_size_fractions[1])])
        self.my_screen = pygame.Surface(self.my_screensize)


        # Todo: change these so that the user can give (l,r,u,d) bounds for plotting
        # OR origin, zoom, relative_scale_y_fraction (which gets internally converted)
        # and an independent range of values that locate the force and sample points.
        self.default_origin_fractions = (.5,.3) # (.5,.5) would be at the center of pane

        # Left, Right, Bottom, and Top bounds for plotting. Default: [-1, 1, 1.4, 0.6]
        if 'LRBT_bounds' in kwargs.keys():
            if 'origin_fractions' in kwargs:
                raise Exception("don't provide both origin center fractions and LRBT bounds")
            self.LRBT_bounds = kwargs['LRBT_bounds']
            self.origin_fractions = (abs(self.LRBT_bounds[0]/(self.LRBT_bounds[1]-self.LRBT_bounds[0])),\
                                     abs(self.LRBT_bounds[2]/(self.LRBT_bounds[2]-self.LRBT_bounds[3])))
            print(self.origin_fractions, (self.LRBT_bounds[0]/(self.LRBT_bounds[1]-self.LRBT_bounds[0])\
                    , self.LRBT_bounds[2]/(self.LRBT_bounds[2]-self.LRBT_bounds[3])))
            _=input()
        else:
            self.origin_fractions = kwargs.get('origin_fractions', self.default_origin_fractions) # (.5,.3)
            default_LRBT_bounds = [-2*self.origin_fractions[0], 2*(1-self.origin_fractions[0]),\
                                   -2*self.origin_fractions[1], 2*(1-self.origin_fractions[1])]
            if 'zoom' in kwargs.keys():
                self.LRBT_bounds = [bound*kwargs['zoom'] for bound in default_LRBT_bounds]
            else:
                self.LRBT_bounds = default_LRBT_bounds 

        self.origin_coords = np.array([int(self.my_screensize[i] * \
                self.origin_fractions[i]) for i in [0,1]])

        ''' Values_width_range: the range of values to calculate functions. 
            Matters most for, eg, e^x-x. '''
        self.values_width_range = kwargs.get('values_width_range', np.array([-5,5]))
        ''' Not checked: origin coords are halfway between values range! '''


        self.color_bit = True if 'bg_color' in kwargs else False
        self.bg_color = kwargs.get('bg_color', self.get_random_bg_color())

        self.nbuckets = 2+1
#        self.nsamples = self.nbuckets *500
        self.nsamples = 250

#        self.create_id_line_points()
#        self.create_exp_line_points()
        self.create_zero_line_points()
#        self.create_exMx_line_points()
        self.create_sample_points_exMx()

#        return

        num_samples = int(self.nsamples *2  ) # empirical - this get approx self.nsamples-many ones within exMx bounds
        self.sample_points_exMx = np.random.uniform(\
                   *self.values_width_range, (num_samples,2))
#        self.sample_points_exMx = np.random.uniform(\
#                   -5, 5, (num_samples,2))
#        self.sample_points_exMx = np.random.uniform(\
#                   -5, 5, (self.nsamples,2))
        self.sample_points_exMx = self.sample_points_exMx[np.where(\
                self.sample_points_exMx[:,1]>0)]
        def exMx(q): return np.exp(q)-q
        self.sample_points_exMx = self.sample_points_exMx[np.where(\
                self.sample_points_exMx[:,1]<exMx(self.sample_points_exMx[:,0]))]
        self.sample_points_exMx *= self.my_screensize / np.log(self.values_width_range[1]) / 3
        self.sample_points_exMx  = self.sample_points_exMx  .astype(int)

        # sample points meant to just fit on the screen
        self.simple_sample_points2 = np.random.uniform(\
                int(-SCREENSIZE[0]*self.origin_fractions[0]) , \
                int(SCREENSIZE[0]*self.origin_fractions[0]), (3,2))
        self.simple_sample_points = np.random.uniform(0,10, (13,2))
#        print(self.simple_sample_points[:3,:])
        self.simple_sample_points *= self.my_screensize / (self.values_width_range[1] *3)
        self.simple_sample_points -= self.origin_coords        
#        print(self.simple_sample_points[:3,:])
##        self.simple_sample_points *= self.my_screensize / self.values_width_range[1] * 3
#        print(self.simple_sample_points[:3,:])
#        _=input(self.simple_sample_points)

        self.simple_zero_line = np.stack([ np.linspace(*self.values_width_range,5), np.zeros((5,))]).T
        self.simple_zero_line = np.stack([ np.linspace(*[i for i in [-100,100]],5), np.zeros((5,))]).T


        self.test_sample_points = np.array([[ 1,1 ],[.3,2 ],[.7,.7],[-1.2,.2]])
        self.test_force_points = np.array([[1,0],[-1,0]])
        print(self.test_sample_points)



    def create_sample_points_exMx(self):
        num_samples = int(self.nsamples *2  ) # empirical - this get approx self.nsamples-many ones within exMx bounds
        self.sample_points_exMx = np.random.uniform(\
                   -5, 5, (num_samples,2))
        self.sample_points_exMx = self.sample_points_exMx[np.where(\
                self.sample_points_exMx[:,1]>0)]
        def exMx(q): return np.exp(q)-q
        self.sample_points_exMx = self.sample_points_exMx[np.where(\
                self.sample_points_exMx[:,1]<exMx(self.sample_points_exMx[:,0]))]

#        print(self.sample_points_exMx)
        self.sample_points_exMx *= self.my_screensize / self.values_width_range[1] / 3
        self.sample_points_exMx  = self.sample_points_exMx  .astype(int)


    ''' ------------------------------------------------------------  '''
    ''' The following helpers create line and curve collections of    '''
    ''' unmovable points which project force.                         '''
    ''' ------------------------------------------------------------  '''

    def create_points(self, func, scale_to_window_width, howmany, buffer_=0):
        x_vals = np.linspace( self.values_width_range[0] - buffer_, 
                              self.values_width_range[1] + buffer_, 
                              howmany )
        points = np.stack([ x_vals, func(x_vals) ]).T
#        print(points)
        points2 = points * self.my_screensize / scale_to_window_width
        return points2
        #return points.astype(int)

    def create_exMx_line_points(self): # exMx: (e^x) - x
        self.exMx_line_points = self.create_points( \
                lambda x:np.exp(x)-x, \
                np.log(self.values_width_range[1]) / 3.,
                self.nbuckets*3)

    def create_zero_line_points(self):
        self.zero_line_points = self.create_points( \
                lambda x:0*x, \
                self.values_width_range[1] * 2.,
                self.nbuckets)

    def create_exp_line_points(self):
        self.exp_line_points = self.create_points( \
                lambda x:np.exp(x), \
#                np.log(self.values_width_range[1][0]) * 3., \
                self.values_width_range[1] * 4., \
                self.nbuckets, \
                buffer_=0)#-2)

    def create_id_line_points(self):
        self.id_line_points = self.create_points( \
                lambda x:x, \
#                self.values_width_range[1][0] * 3., \
                self.values_width_range[1] * 4., \
                self.nbuckets)




    def get_random_bg_color(self):
        return (random.randrange(224,255), \
                random.randrange(224,255), \
                random.randrange(224,255))
#        return (random.randrange(192,255), \
#                random.randrange(192,255), \
#                random.randrange(192,255))

    ''' ------------------------------------------------------------  '''
    ''' The following helpers implement the various force functions.  '''
    '''                                                               '''
    ''' ------------------------------------------------------------  '''


    ''' stopping small values from blasting off is very unnecessary. '''
    def apply_inverse_dist_squared_force__third_try(self):  
        self.simple_sample_points *= self.values_width_range[1] * 3 / self.my_screensize 
        self.test_sample_points
        self.test_force_points 


    def apply_inverse_dist_squared_force__second_try(self):  
        # the first issue is clearly in developing cross_array
#        cross_array=\
#                np.add(-self.sample_points_exMx.T, \
#                np.reshape(self.zero_line_points.T,\
#                self.zero_line_points.shape+(1,)) \
#                )#.shape )
        self.simple_sample_points *= self.values_width_range[1] * 3 / self.my_screensize 
        cross_array=\
                np.add( \
                self.simple_zero_line ,\
                -np.expand_dims(self.simple_sample_points, 1)\
                )#.shape )
#        cross_array=\
#                np.add( \
#                self.zero_line_points ,\
#                -np.expand_dims(self.simple_sample_points, 1)\
#                )#.shape )
        # ^ check!
        axes_names = AX_SAMPLES, AX_FORCES, AX_2D = 0,1,2
        magn = np.power(np.sum(np.power(cross_array,2),axis=AX_2D),0.5)
        event_horizon_mask = np.where(magn < epsilon_touching) # if a point is too close, don't let it catapult out. 
        mask_=np.ones(self.simple_sample_points.shape)
        mask_[(event_horizon_mask[1],)]=0
        magn = np.minimum(magn, 100.0)
        ''' Broadcasting math. A has shape (i=a,j=b,k=c) and B has shape (i=a,j=b). 
            That is, A[i=0,j=b-1,k=c/2] is an index. To get each A[:,:,k] elementwise
            divided by B[:,:], that is C_ijk = [A_ij/B_ij]{1...c}, do:
            np.swapaxes(np.swapaxes(A,0,2) / np.swapaxes(B,0,1)==B.T, 0,2)
        '''
        unit_vectors =  np.swapaxes(np.swapaxes(cross_array, AX_SAMPLES,AX_2D) \
                      / np.swapaxes(magn,AX_SAMPLES,AX_FORCES), AX_2D,AX_SAMPLES)
        inv_dist_vecs = np.swapaxes(np.swapaxes(unit_vectors,0,2) / np.swapaxes(magn,0,1), 0,2)
        inv_dist_sqr_vecs = np.swapaxes(np.swapaxes(inv_dist_vecs ,0,2) / np.swapaxes(magn,0,1), 0,2)
        accum_normalized_vectors = np.sum(inv_dist_sqr_vecs , axis=1)
        self.simple_sample_points += accum_normalized_vectors *0.1
        self.simple_sample_points *= self.my_screensize / self.values_width_range[1] / 3
        return



    def apply_inverse_dist_squared_force(self):  
        cross_array=\
               np.add(self.sample_points_exMx.T, \
                np.reshape(self.zero_line_points.T,\
                self.zero_line_points.shape+(1,)) \
                )#.shape )
        magn = np.power(np.sum(np.power(cross_array,2),axis=0),0.5)
        if np.power(np.sum(np.power(magn,2)),0.5)<epsilon_touching: return
        avg_magn = np.mean(magn, axis=1)
        magn = (magn.T/np.max(magn,axis=1)).T
        inv_dist_sqrs = cross_array / magn
        accums=np.sum(inv_dist_sqrs , axis=0)
        self.sample_points_exMx = self.sample_points_exMx - accums.T.astype(np.float64)*0.001

    def apply_unit_force(self): # subgradient-style: add all distanced-strengthed vectors then normalize
        self.sample_points_exMx 
        cross_array=\
               np.add(self.sample_points_exMx.T, \
                np.reshape(self.zero_line_points.T,\
                self.zero_line_points.shape+(1,)) \
                )#.shape )
        accums=np.sum(cross_array, axis=0)
        normz_accums = accums / np.power(np.sum(np.power(accums,2),axis=0),0.5)
        self.sample_points_exMx = self.sample_points_exMx - normz_accums.T.astype(np.float64)*20

    ''' ------------------------------------------------------------  '''
    ''' The following helpers collect the steps for rendering points  '''
    ''' and various plotting settings and options.                    '''
    ''' ------------------------------------------------------------  '''

    def draw_me(self):
        self.my_screen.fill(self.bg_color)
        if not self.color_bit:
            self.color_bit=True
            self.bg_color = self.get_random_bg_color() 
        self.plot_points_current(self.test_sample_points, size=1, color=(0,0,0))
        self.plot_points_current(self.test_force_points, size=1, color=(0,0,0))
        self.plot_points_old(np.array([[0,0]]), (127,127,0),1)
        pygame.Surface.blit(self.global_screen, self.my_screen, self.pane_location)

    def plot_points_current(self, points, color=(0,0,255), size=1):
        pygame.draw.polygon(self.my_screen, (0,0,0), \
                ((0,0),\
                 (self.my_screensize[0],self.my_screensize[1]-1),\
                 (self.my_screensize[0],self.my_screensize[1]),\
                 (0,1)))
        bds=self.LRBT_bounds[:]
        bds[0]*=self.my_screensize[0]
        bds[1]*=self.my_screensize[0]
        bds[2]*=self.my_screensize[1]
        bds[3]*=self.my_screensize[1]
        ops=self.origin_coords
        pygame.draw.line(self.my_screen, (255,0,255),\
                (bds[0],self.my_screensize[1]-ops[1]),\
                (bds[1],self.my_screensize[1]-ops[1]) )
        pygame.draw.line(self.my_screen, (0,255,255),\
                (ops[0],bds[2]),\
                (ops[0],bds[3]) )

        # tag [62362]

        screen_width  = self.LRBT_bounds[1] - self.LRBT_bounds[0]
        screen_height = self.LRBT_bounds[3] - self.LRBT_bounds[2]
        scale_x = self.my_screensize[0]/screen_width  
        scale_y = self.my_screensize[1]/screen_height   

        print('    bounds:', self.LRBT_bounds[:],'->',bds)
        print('    origin coords:', self.origin_coords)

        pygame.font.init() 
        myfont = pygame.font.SysFont('Comic Sans MS', 10)

        origin_coords_2=np.array([0,self.my_screensize[1]])-self.origin_coords*np.array([-1,1])
        print('    origin coords2:', origin_coords_2)
        print('    screen size:', self.my_screensize)
        pygame.draw.circle(self.my_screen, (127,255,127), \
                origin_coords_2,4)
        textsurface = myfont.render('0,0', False, (0, 255, 0))
        self.my_screen.blit(textsurface, origin_coords_2)

        ''' plot a point every 100 pixels: '''
        range_ = [100*z for z in range(self.my_screensize[0]//100+1)]
        for x in range_:
            for y in range_:
                pygame.draw.circle(self.my_screen, (255,192,0),(x,y),2)

        '''
self.LRBT_bounds =  [-1, 1, -0.6, 1.4] < > v ^
width, height = 2,2
screensize=360x360
origin_coords_2 = 180,252  <- add these at the very end
[0.9,0.9] -> 162,162  ==  (360/2)*.9 == screensize / width * value
          -> 324,90 == (origin_coords_x + points_x, origin_coords_y - points_y)

        '''

        for i, (x0,y0) in enumerate(np.array([[.9,.9],[-.9,-.4],[.1,.9]])):
            print(x0,y0,'->',end=' ')
            x = origin_coords_2[0]+x0*scale_x
            y = origin_coords_2[1]-y0*scale_y
            print(x,y)
            n=10
            pygame.draw.polygon(self.my_screen, (255,255,255), \
                ((x+n,y+n),(x-n,y-n),(x+n,y-n),(x-n,y+n)))
            textsurface = myfont.render(str((x0,y0)), False, (0, 255, 0))
            self.my_screen.blit(textsurface, (x,y))
        return
        for (x,y) in points:#+self.origin_coords:
            print(x,y,'->',end=' ')
            # incomplete
            x = (x * self.my_screensize[0]) - self.origin_coords[0]
            y = (y * self.my_screensize[1]) - self.origin_coords[1]
            print(x,y)
            pygame.draw.polygon(self.my_screen, (255,127,0), \
                ((x-size,y-size),(x-size,y+size),(x+size,y+size),(x+size,y-size)))


        for (x,y) in [tuple(points[:,i]) for i in range(points.shape[1])]:#+self.origin_coords:
            x = (x-self.origin_coords[0])*self.my_screensize[0]/2.1+self.origin_coords[0]
            y = (y-self.origin_coords[1])*self.my_screensize[1]/2.1+self.origin_coords[1]
            y=self.my_screensize[1]-y 
            pygame.draw.polygon(self.my_screen, color, \
                ((x-size,y-size),(x-size,y+size),(x+size,y+size),(x+size,y-size)))

    def plot_points_old(self, points=[], color=(0,0,255), size=0):
        if len(points)==0:
            points = np.array([ [2,-3], [5,4] ])
            points = np.array([ [-6,-6], [2,-3], [5,4] ])
            points = np.array([ [0,0], [0,4], [3,4] ])
            points = np.array([ [-2,-4], [2,4], [-2,4] ])
        for (x,y) in points+self.origin_coords:
            y=self.my_screensize[1]-y 
            pygame.draw.polygon(self.my_screen, color, \
                ((x-size,y-size),(x-size,y+size),(x+size,y+size),(x+size,y-size)))
#                    ((x-1,y-1),(x-1,y+1),(x+1,y+1),(x+1,y-1)))
       

            

class Particle(object):
    pass

class IdLine(object):
    def __init__(self, origin, rotation):
        pass
class ExpLine(object):
    def __init__(self, origin, rotation):
        pass

pygame.init()
screen = pygame.display.set_mode(SCREENSIZE)
ORIGIN = tuple([int(s/2) for s in SCREENSIZE])

# pane instantiations
panes = [Pane(screen, bg_color=(255,255,224))]
''' do the following in conjunction with :
SCREENSIZE = (120*6, 120*3) # 120 is nice and divisible'''
panes=[Pane(screen, pane_coords_fractions=(0,0), pane_size_fractions=(1/3.,1), values_width_range=[-8,8], LRBT_bounds=[-4,4,-4,4]),\
       Pane(screen, pane_coords_fractions=(1/3.,0), pane_size_fractions=(1/3.,1), values_width_range=[-8,8], LRBT_bounds=[-4,7,-2,2]),\
       Pane(screen, pane_coords_fractions=(2/3.,0), pane_size_fractions=(1/3.,1), values_width_range=[-1,1], LRBT_bounds=[-1,1,-0.6,1.4])]
exit_bit=False
Events=[]
fps=1

while True:
    Events = pygame.event.get()
    for e in Events:
        if e.type == pygame.KEYDOWN:
            if e.key==pygame.K_SPACE: 
                time.sleep(0.2)
                pygame.event.clear()
                pygame.display.set_caption("PAUSED fps cap: " + str(1./fps))
                e2=pygame.event.wait()
                pygame.display.set_caption("RESUMING fps cap: " + str(1./fps))
                continue
            if e.key == pygame.K_UP:
                fps *= 0.5
            elif e.key == pygame.K_DOWN:
                fps = min(1,fps*2)
            else: exit_bit=True
        if e.type == pygame.QUIT:
            exit_bit=True
    if exit_bit: break

    time.sleep(fps)
    screen.fill(pygame.Color(150,200,200))
    for pane in panes:
        pane.draw_me()
    pygame.display.flip()
    pygame.display.set_caption("fps cap: " + str(1./fps))
    if fps>0.001:print('-')
    for pane in panes:
        pane.apply_inverse_dist_squared_force__third_try()
    
    if input():break

time.sleep(0.5)
pygame.quit()
#        import sys; _=input(); sys.exit()
