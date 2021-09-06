import pymunk
import pygame
import pymunk.pygame_util
import random
import numpy as np

SCREENSIZE = (600,600)
touching_epsilon = 1e-3
force_line_bound = int( 600/2 * 0.9 )
n_buckets = 140



def get_magn(vec, kind='eucl'): return (vec[0]**2+vec[1]**2)**0.5
def mult(vec, m): return [vec[0]*m, vec[1]*m]
def to_ints(vec): return (int(vec[0]), int(vec[1]))
def cap_values(vec, abs_val): 
    if type(vec)==iter:
        return [min(max(vec[0], -abs_val), abs_val), \
                min(max(vec[1], -abs_val), abs_val)]
    else:
        try:
            _= min(max(vec, -abs_val), abs_val)
            return _
        except: raise Exception(vec, type(vec), abs_val)

class Particle(object):
    def __init__(self, **kwargs):
        self.xpos, self.ypos, self.xvel, self.yvel = 0,0,0,0
        self.kind=kwargs['kind']
        if self.kind=='random_mobile':
            force_carrying=True
            force_subjected=True
            self.xpos = random.randrange(0,SCREENSIZE[0])
            self.ypos = random.randrange(0,SCREENSIZE[1])
        elif self.kind=='forceless_test_point':
            force_carrying=False
            force_subjected=True
            self.xpos, self.ypos = kwargs['init_pos']
        elif self.kind=='fixed_at_center':
            force_carrying=True
            force_subjected=True
            self.xpos = int(SCREENSIZE[0]/2)
            self.ypos = int(SCREENSIZE[1]/2)
        elif self.kind=='force_line':
            force_carrying=True
            force_subjected=False
            self.xpos, self.ypos = kwargs['init_pos']
        self.radius = kwargs.get('radius', 3)
        if not type(self.radius)==int:
            print('warning: radius not int')
#        self.screen = kwargs.get('screen', pygame.Surface((0,0))) # a Surface
        self.screen = kwargs['screen']
        if type(kwargs.get('color'))==str:
            self.color = pygame.Color(kwargs.get('color'))
        else:
            self.color = kwargs.get('color', pygame.Color('blue'))
        self.position=self.pos = [self.xpos, self.ypos]
        self.velocity=self.vel = [self.xvel, self.yvel]

    def register_bodies(self, bodies):
        self.bodies=bodies

    def draw_me(self):
        # pymunk/pymunk/pygame_util.py
        try:
            pygame.draw.circle(self.screen, self.color, to_ints(self.pos), self.radius)
        except:
            raise Exception(self.pos, self.vel, self.radius)
#       help(pygame.draw.circle)
#          circle(...)
#              circle(Surface, color, pos, radius, width=0) -> Rect
#              draw a circle around a point

    def sub(self, body): return [body.xpos-self.xpos, body.ypos-self.ypos]
    def accum_forces(self, force_kind='inverse_distance_squared', cap=None, inversion_pivot=1):
        if self.kind in ('fixed_at_center', 'force_line'): return
        bodies=self.bodies
        accum_x, accum_y = 0.,0.
        for body in bodies:
            if body==self: continue
            if body.kind in ('forceless_test_point',): continue
            diff_vec = self.sub(body)
            magn = get_magn(diff_vec)
            if magn < touching_epsilon: continue
            normz_vec = mult(diff_vec, magn**-1)
            force_applied = [0,0]
            if force_kind=='inverse_distance_squared':
                force_applied = mult(normz_vec, inversion_pivot/magn**2)
#                if cap:
#                    force_applied = cap_values(force_applied, cap)
            elif force_kind=='unit':
                force_applied = mult(normz_vec, 1)
            elif force_kind=='inverse_distance':
                force_applied = mult(normz_vec, inversion_pivot/magn)
            elif force_kind=='exponential':
                force_applied = mult(normz_vec, 2**min(1, magn)/2)
            else:
                raise Exception(force_kind)
            accum_x += force_applied[0]
            accum_y += force_applied[1]
        self._accum_forces = [accum_x,accum_y]
        return self._accum_forces

    def move_via_accum_forces(self, multiplier=1, limit=0.5, to_int=True):
        if self.kind in ('fixed_at_center', 'force_line'): return
        delta = self._accum_forces
        if not limit=='no limit': 
            delta[0] = cap_values(delta[0], limit)
            delta[1] = cap_values(delta[1], limit)
        self.xpos += delta[0]*multiplier
        self.ypos += delta[1]*multiplier
        if to_int: self.xpos=int(self.xpos)
        if to_int: self.ypos=int(self.ypos)
        self.position=self.pos = [self.xpos, self.ypos]
        return 200

    def move_via_velocity(self, multiplier=1, to_int=True, gamma=0.0, limit=0.5):
      try:
        if self.kind in ('fixed_at_center', 'force_line'): return
        delta = self._accum_forces
#        self.xvel += delta[0]*(1-gamma) + self.xvel*gamma
#        self.yvel += delta[1]*(1-gamma) + self.yvel*gamma
        self.xvel += delta[0]#*(1-gamma) + self.xvel*gamma
        self.yvel += delta[1]#*(1-gamma) + self.yvel*gamma
        if self.kind in ('fixed_at_center', 'force_line'):
            self.xvel, self.yvel = 0,0
        if not limit=='no limit': 
            self.xvel = cap_values(self.xvel, limit)
            self.yvel = cap_values(self.yvel, limit)
        if not (limit/2.3 or limit=='no limit'):
            raise Exception(limit)
        self.xpos += self.xvel*multiplier
        self.ypos += self.yvel*multiplier
        self.position=self.pos = [self.xpos, self.ypos]
        if to_int: 
            self.xpos=int(self.xpos)
            self.xvel=int(self.xvel)
            self.ypos=int(self.ypos)
            self.yvel=int(self.yvel)
        self.velocity=self.vel = [self.xvel, self.yvel]
        self.position=self.pos = [self.xpos, self.ypos]
        return 200
      except: 
        raise Exception(delta, self._accum_forces, self.pos, self.vel, self.xvel, self.kind)



def sub(b,a):
    assert(type(a)==Particle and type(b)==Particle)




pygame.init()
screen = pygame.display.set_mode(SCREENSIZE)
clock = pygame.time.Clock()

bodies=[]
#for n in np.linspace(-force_line_bound, force_line_bound, n_buckets):
exp_min, exp_max = -15, 5
exp_range = np.linspace(exp_min, exp_max, n_buckets)
for ni, n in enumerate(np.linspace(-1000, 1000+SCREENSIZE[0], n_buckets)):
    center = 2 # a factor that shifts the center. 2 = centered at middle. 1 = the 2^0=1 is at far left.
    shift_up = 60
    scaling = 15
    top=SCREENSIZE[0];
#    p = Particle(kind='force_line', screen=screen, init_pos=[n,0.9*SCREENSIZE[1]])
    p = Particle(kind='force_line', screen=screen, init_pos=[n,top-shift_up], color='purple')
    bodies.append(p)
#    p = Particle(kind='force_line', screen=screen, init_pos=[n,top-n])
#    bodies.append(p)
#    e_to_the_n = top- 2**( (n-top/center) / (np.log2(top)*2) )
#    e_to_the_n = (top- 2**( (n-top/center) / scaling )) - top/center
#    e_to_the_n_minus_n = e_to_the_n - (top/center - n)
    e_to_the_n =  ( top - top*2**(exp_range[ni]))
    p = Particle(kind='force_line', screen=screen, color='pink', init_pos=[n,e_to_the_n-shift_up ])
    p = Particle(kind='force_line', screen=screen, color='red' , init_pos=[n,e_to_the_n - (top/center-n) - shift_up ])
#    p = Particle(kind='force_line', screen=screen, init_pos=[n,2**exp_range[ni]-exp_range[ni] ], color='yellow')
    bodies.append(p)
    if 0 and not ni%6:
        p = Particle(kind='force_line', screen=screen, color='magenta', init_pos=[n,e_to_the_n ])
        bodies.append(p)

m_buckets = 30
for m1 in  np.linspace(0, SCREENSIZE[0], m_buckets):
    for m2 in  np.linspace(0, SCREENSIZE[0], m_buckets):
        bodies += [Particle(kind='forceless_test_point', screen=screen, color='black', \
                    init_pos=[m1+random.random()*4-2, m2+random.random()*4-2])]
#for _ in range(2):
#    m1 = random.random()*SCREENSIZE[0]/2
#    t=100
#    while 1:
#        m2 = random.random()*SCREENSIZE[0]
#        if m2 < top - top*2**m1:
#            break
#        t-=1
#        if t==0: raise Exception()
#    bodies += [Particle(kind='forceless_test_point', screen=screen, color='blue', \
#                    init_pos=[m1, m2])]

#bodies = [Particle(kind='forceless_test_point', screen=screen, init_pos=[300,250]), \
#          Particle(kind='forceless_test_point', screen=screen, init_pos=[250,250])]
#bodies += [Particle(kind='random_mobile', screen=screen, color='black', \
#        init_pos=[random.randrange(0,SCREENSIZE[0]), random.randrange(0,SCREENSIZE[0])]) for _ in range(50)]

#bodies += [Particle(kind='random_mobile', screen=screen) for _ in range(44)]
#bodies += [Particle(kind='fixed_at_center', screen=screen)]
for body in bodies:
    body.register_bodies(bodies)
screen.fill(pygame.Color("pink"))
for b in bodies:
    b.draw_me()

pygame.draw.polygon(screen, (0,255,0), ((125,125),(375,100),(375,300),(125,275)))

e=False
#fps=2
#for t in range(10):
#    print('TICK '+str(t))
fps=800
while True:
#    print('bodies:')
#    print(bodies)
#    print()
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            e=True
            break
    if e: break
#    for event in pygame.event.get():
#        if (
#            event.type == pygame.QUIT
#            or event.type == pygame.KEYDOWN
#            and event.key == pygame.K_ESCAPE
#        ):
#            exit()

    # 'Star' in the center of screen
#    screen.fill(pygame.Color(random.randrange(100,200),200,200))
    screen.fill(pygame.Color(150,200,200))
    pygame.draw.circle(screen, pygame.Color("yellow"), (300, 300), 10)
    for body in bodies:
        body.accum_forces(force_kind='inverse_distance_squared', inversion_pivot=100)
#        body.accum_forces(force_kind='exponential')
    for body in bodies: 
#        body.move_via_accum_forces(multiplier=4, to_int=False) # not 'force' 2nd deriv
        body.move_via_velocity(multiplier=1, limit=10, to_int=False) 
    for body in bodies: 
        body.draw_me()

#    space.debug_draw(draw_options)

#    for body in bodies:
#        body.velocity += pymunk.Vec2d(random.randint(5, 50), \
#                                     random.randint(5, 50))
#    pygame.display.flip()
    pygame.display.set_caption("fps: " + str(clock.get_fps()))
#    space.step(1. / fps)
    clock.tick(fps)
