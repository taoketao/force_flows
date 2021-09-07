

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
#        self.pane_coords_fraction = kwargs.get('pane_coords_x',0), \
#                                    kwargs.get('pane_coords_y',0)
        self.pane_coords_fractions = kwargs.get('pane_coords_fractions', (0,0))
        self.pane_location = tuple([int(SCREENSIZE[i]*self.pane_coords_fractions[i]) for i in [0,1]])
#        self.pane_fraction_x, self.pane_fraction_y = \
#                kwargs.get('pane_fraction_x', 1), \
#                kwargs.get('pane_fraction_y', 1);
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
        self.bg_color = kwargs.get('bg_color', self.get_random_bg_color())

        nbuckets = 40+1
        self.id_line_points = np.array([ [i,i] for i in np.linspace(\
                   self.values_range[0][0], self.values_range[1][0], nbuckets  ) ])
        self.id_line_points *= self.my_screensize / self.values_range[1][0] / 2
        self.id_line_points = self.id_line_points.astype(int)
                
    def get_random_bg_color(self):
        return (random.randrange(192,255), \
                random.randrange(192,255), \
                random.randrange(192,255))

    def draw_me(self):
        self.my_screen.fill(self.bg_color)
        self.bg_color = self.get_random_bg_color() # for now
        self.plot_points('id_line')
        self.plot_points(np.array([[0,0]]), (255,127,0))
        pygame.Surface.blit(self.global_screen, self.my_screen, self.pane_location)

    def plot_points(self, points=None, color=(0,0,255)):
        # tmp:
        if points==None:
            points = np.array([ [2,-3], [5,4] ])
            points = np.array([ [-6,-6], [2,-3], [5,4] ])
            points = np.array([ [0,0], [0,4], [3,4] ])
            points = np.array([ [-2,-4], [2,4], [-2,4] ])
        if points=='id_line':
            points = self.id_line_points
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
            print( x,y)
            pygame.draw.polygon(self.my_screen, color, \
                    ((x-1,y-1),(x-1,y+1),(x+1,y+1),(x+1,y-1)))
       

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
panes = [Pane(screen)]
panes=[Pane(screen, pane_coords_fractions=(0,0), pane_size_fractions=(0.4,0.2)),\
      Pane(screen, pane_coords_fractions=(0.4,0), pane_size_fractions=(0.6,0.4)),\
      Pane(screen, pane_coords_fractions=(0,0.2), pane_size_fractions=(0.4,0.8)),\
      Pane(screen, pane_coords_fractions=(0.4,0.2),pane_size_fractions=(0.6,0.8))]
panes = [Pane(screen, pane_coords_fractions=(i/3., j/3.),\
                       pane_size_fractions=(1/.3, 1/.3)) \
        for i in range(3) for j in range(3)]
panes =\
    [Pane(screen, pane_coords_fractions=( 0,.0),pane_size_fractions=(.3, .3)),\
     Pane(screen, pane_coords_fractions=(.3,.0),pane_size_fractions=(.3, .3)),\
     Pane(screen, pane_coords_fractions=(.6,.0),pane_size_fractions=(.4, .3)),\
     Pane(screen, pane_coords_fractions=( 0,.3),pane_size_fractions=(.3, .3)),\
     Pane(screen, pane_coords_fractions=(.3,.3),pane_size_fractions=(.3, .3)),\
     Pane(screen, pane_coords_fractions=(.6,.3),pane_size_fractions=(.4, .3)),\
     Pane(screen, pane_coords_fractions=( 0,.6),pane_size_fractions=(.3, .4)),\
     Pane(screen, pane_coords_fractions=(.3,.6),pane_size_fractions=(.3, .4)),\
     Pane(screen, pane_coords_fractions=(.6,.6),pane_size_fractions=(.4, .4))]

while not any([pygame.KEYDOWN == e.type for e in pygame.event.get()]):
    time.sleep(1)
    screen.fill(pygame.Color(150,200,200))
#    pygame.draw.circle(screen, pygame.Color("yellow"), ORIGIN, 10)
    for pane in panes:
        pane.draw_me()
#        pane.plot_points()
    pygame.display.flip()
    print('-')
time.sleep(0.5)
