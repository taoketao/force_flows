

# A vectorized adaptation of my-simulation.py that uses numpy
# vector operations for simulation speed efficiency.

import numpy as np
import random
import pygame
import time

SCREENSIZE = (120*3, 120*3) # 120 is nice and divisible
DEBUG_POINTS_ARITHMETIC = False # flag 

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
        self.origin_fractions = kwargs.get('origin_fractions', (.5,.3)) # (.5,.2)
        self.origin_coords = np.array([int(self.my_screensize[i] * \
                self.origin_fractions[i]) for i in [0,1]])
        self.values_range = kwargs.get('values_range', np.array([[-5,-5],[5,5]]))
        # not checked: origin coords are halfway between values range
        if 'bg_color' in kwargs:
            self.color_bit=True
        self.bg_color = kwargs.get('bg_color', self.get_random_bg_color())

        self.nbuckets = 4+1
        self.nsamples = self.nbuckets *500

        self.create_id_line_points()


        self.zero_line_points = np.array([ [i,0] for i in np.linspace(\
                   self.values_range[0][0], self.values_range[1][1], self.nbuckets  ) ])
        self.zero_line_points *= self.my_screensize / self.values_range[1][0] / 2
        self.zero_line_points = self.zero_line_points.astype(int)
                
        self.exMx_line_points = np.array([ [i,np.exp(i)-i] for i in np.linspace(\
                   self.values_range[0][0], self.values_range[1][1], self.nbuckets*3  ) ])
        self.exMx_line_points *= self.my_screensize / np.log(self.values_range[1][0]) / 3
        self.exMx_line_points = self.exMx_line_points.astype(int)

        self.sample_points_exMx = np.random.uniform(\
                   -5, 5, (self.nsamples,2))
        self.sample_points_exMx = self.sample_points_exMx[np.where(\
                self.sample_points_exMx[:,1]>0)]
        def exMx(q): return np.exp(q)-q
        self.sample_points_exMx = self.sample_points_exMx[np.where(\
                self.sample_points_exMx[:,1]<exMx(self.sample_points_exMx[:,0]))]
        self.sample_points_exMx *= self.my_screensize / np.log(self.values_range[1][0]) / 3
        self.sample_points_exMx  = self.sample_points_exMx  .astype(int)

    def create_points(self, func, scale, howmany, buffer_=0):
        points = np.array([ [i,func(i)] for i in np.linspace(\
                   self.values_range[0][0]-buffer_, self.values_range[1][1]+buffer_, howmany ) ])
        # todo: vectorise ^^
        x_vals = np.linspace(self.values_range[0][0]-buffer_, self.values_range[1][1]+buffer_, howmany )
        points = np.stack([x_vals,func(x_vals)])
        print (points, self.my_screensize, scale)
        points = (points.T * self.my_screensize / scale).T
        print (points)
        input()
        return points#.astype(int)

    def create_id_line_points(self):
        self.id_line_points = self.create_points( lambda x:x, self.values_range[1][0]*3., self.nbuckets)

    def create_exp_line_points(self):
        self.exp_line_points = self.create_points( lambda x:np.exp(x), np.log(self.values_range[1][0]*3.), self.nbuckets, buffer_=-2)
        self.exp_line_points = np.array([ [i,np.exp(i)] for i in np.linspace(\
                    self.values_range[0][0]+2, self.values_range[1][1]-2, self.nbuckets*3  ) ])
        self.exp_line_points *= self.my_screensize / np.log(self.values_range[1][0] )/ 3
        self.exp_line_points = self.exp_line_points.astype(int)


    def get_random_bg_color(self):
        return (random.randrange(192,255), \
                random.randrange(192,255), \
                random.randrange(192,255))

    def draw_me(self):
        self.my_screen.fill(self.bg_color)
        if not self.color_bit:self.bg_color = self.get_random_bg_color() 
        self.plot_points(self.sample_points_exMx, (255,0,0), size=0.5)
        self.plot_points(self.id_line_points)
        self.plot_points(self.exp_line_points)
        self.plot_points(self.zero_line_points, size=1)
        self.plot_points(self.exMx_line_points)
        self.force_lines=(self.zero_line_points, self.exMx_line_points)
        self.plot_points(np.array([[0,0]]), (127,127,0),1)
        pygame.Surface.blit(self.global_screen, self.my_screen, self.pane_location)

    def apply_inverse_dist_squared_force(self):  
        cross_array=\
               np.add(self.sample_points_exMx.T, \
                np.reshape(self.zero_line_points.T,\
                self.zero_line_points.shape+(1,)) \
                )#.shape )
        magn = np.power(np.sum(np.power(cross_array,2),axis=0),0.5)
        print (magn)
        avg_magn = np.mean(magn, axis=1)
        magn = (magn.T/np.max(magn,axis=1)).T
        print (magn)
        print(magn.shape, avg_magn.shape, avg_magn)
        inv_dist_sqrs = cross_array / magn
        print(inv_dist_sqrs[:3,:3,:3] )
        inv_dist_sqrs /= magn
#        print(inv_dist_sqrs[:3,:3,:3] )
        accums=np.sum(inv_dist_sqrs , axis=0)
        print(accums[:3,:3])
        print('---')
#        self.sample_points_exMx -= accums.T


    def apply_unit_force(self): # subgradient-style: add all distanced-strengthed vectors then normalize
        self.sample_points_exMx 
        print(self.sample_points_exMx.shape, self.zero_line_points.shape )
        cross_array=\
               np.add(self.sample_points_exMx.T, \
                np.reshape(self.zero_line_points.T,\
                self.zero_line_points.shape+(1,)) \
                )#.shape )
        accums=np.sum(cross_array, axis=0)
        print (accums)
        normz_accums = accums / np.power(np.sum(np.power(accums,2),axis=0),0.5)
        print (normz_accums)
        self.sample_points_exMx -= normz_accums.T*20

    def plot_points(self, points=[], color=(0,0,255), size=0):
        # tmp:
        if len(points)==0:
            points = np.array([ [2,-3], [5,4] ])
            points = np.array([ [-6,-6], [2,-3], [5,4] ])
            points = np.array([ [0,0], [0,4], [3,4] ])
            points = np.array([ [-2,-4], [2,4], [-2,4] ])
        if DEBUG_POINTS_ARITHMETIC: 
            print( points, self.values_range )
            print( [(self.values_range[i]-points).flatten() for i in [0,1]])
            print( [(points-self.values_range[i]).flatten() for i in [0,1]])
        if False and not ((np.all(self.values_range[1]-points >= 0) and\
                np.all(points-self.values_range[0] >= 0))):
            raise Exception("points can't be out of window bounds/not implemented")
            pass#  todo: instead, just don't *plot* these points, but \
            pass#  include them in calculations with caveat message
        if DEBUG_POINTS_ARITHMETIC: 
            print(points + self.origin_coords)
            print(self.origin_coords)
        
        for (x,y) in points+self.origin_coords:
            y=self.my_screensize[1]-y 
            if DEBUG_POINTS_ARITHMETIC: print( x,y)
            pygame.draw.polygon(self.my_screen, color, \
                ((x-size,y-size),(x-size,y+size),(x+size,y+size),(x+size,y-size)))
#                    ((x-1,y-1),(x-1,y+1),(x+1,y+1),(x+1,y-1)))
       

#        x_value, y_value = points
#        x_plot_coord = x_value - self.origin

    def draw_lines(self):
        plot_points
        buckets = list(range(-10,10+1))
        x_diff = len(buckets)-1
        for b in buckets:
            pass
            

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

# panes definitions
panes=[Pane(screen, pane_coords_fractions=(0,0), pane_size_fractions=(0.4,0.2)),\
      Pane(screen, pane_coords_fractions=(0.4,0), pane_size_fractions=(0.6,0.2)),\
      Pane(screen, pane_coords_fractions=(0,0.2), pane_size_fractions=(0.4,0.8)),\
      Pane(screen, pane_coords_fractions=(0.4,0.2),pane_size_fractions=(0.6,0.8))]
panes = [Pane(screen, bg_color=(255,255,224))]
#panes = [Pane(screen, pane_coords_fractions=(i/3., j/3.),\
#                       pane_size_fractions=(1/.3, 1/.3)) \
#        for i in range(3) for j in range(3)]
#panes =\
#    [Pane(screen, pane_coords_fractions=( 0,.0),pane_size_fractions=(.3, .3)),\
#     Pane(screen, pane_coords_fractions=(.3,.0),pane_size_fractions=(.3, .3)),\
#     Pane(screen, pane_coords_fractions=(.6,.0),pane_size_fractions=(.4, .3)),\
#     Pane(screen, pane_coords_fractions=( 0,.3),pane_size_fractions=(.3, .3)),\
#     Pane(screen, pane_coords_fractions=(.3,.3),pane_size_fractions=(.3, .3)),\
#     Pane(screen, pane_coords_fractions=(.6,.3),pane_size_fractions=(.4, .3)),\
#     Pane(screen, pane_coords_fractions=( 0,.6),pane_size_fractions=(.3, .4)),\
#     Pane(screen, pane_coords_fractions=(.3,.6),pane_size_fractions=(.3, .4)),\
#     Pane(screen, pane_coords_fractions=(.6,.6),pane_size_fractions=(.4, .4))]

#while not any([pygame.KEYDOWN == e.type for e in pygame.event.get()]):
exit_bit=False
Events=[]
#while not any([pygame.KEYDOWN == e.type for e in Events]):
#while not any([pygame.KEYDOWN == e.type and not pygame.K_SPACE == e.key for e in Events]):
fps=1
while True:
    Events = pygame.event.get()
#    if any([pygame.KEYDOWN == e.type and pygame.K_SPACE == e.key for e in Events]):
#        break
    for e in Events:
        if e.type == pygame.KEYDOWN:
            if e.key==pygame.K_SPACE: 
                time.sleep(0.2)
                pygame.event.clear()
                pygame.display.set_caption("PAUSED fps cap: " + str(1./fps))
                e2=pygame.event.wait()
                pygame.display.set_caption("RESUMING fps cap: " + str(1./fps))
                continue
#                if e2.type == pygame.KEYDOWN and e2.key == pygame.K_SPACE:
#                    continue
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
#    pygame.draw.circle(screen, pygame.Color("yellow"), ORIGIN, 10)
    for pane in panes:
        pane.draw_me()
#        pane.plot_points()
    pygame.display.flip()
    pygame.display.set_caption("fps cap: " + str(1./fps))
    print('-')
#    if any([e.type == pygame.KEYDOWN and pygame.K_SPACE == e.key \
#                for e in pygame.event.get()]):
#        event = pygame.event.wait()
    for pane in panes:
        pane.apply_inverse_dist_squared_force()
time.sleep(0.5)
pygame.quit()
