import numpy as np
import numpy.random as nr
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm

def strat1(arena_odor, starting_position_x, starting_position_y, odor_radius, fly_sensitivity = 0.10, refresh_rate = 0.10, arena_x=100, arena_y=20, num_trials=10000, fly_time=2500, save=False):

   # Things to return
   ss1_num_successes = 0
   ss1_num_completed_trials = 0

   # Arena distribution 
   arena_fly_start_y = arena_y / 2
   arena_fly_start_x = arena_x - 1
   
   # Search strategy 1
   # fly trajectory
   for j in tqdm(range(num_trials)):
      xs = np.zeros(fly_time)
      ys = np.zeros(fly_time)

      xs[0] = arena_fly_start_x
      ys[0] = arena_fly_start_y

      # Now generating the random walk
      for i in range(1, fly_time):
         illegal_move = True

         # Generating the probability
         while illegal_move:
            rnum = nr.rand()

         # Testing to see if an illegal move and the 
         # random num needs to be generated again
            if rnum < 0.25 and ys[i-1] + 1 < arena_y: 
               illegal_move = False
            elif rnum >= 0.25 and rnum < 0.5 and ys[i-1] - 1 > 0:
               illegal_move = False
            elif rnum >= 0.5 and rnum < 0.75 and xs[i-1] - 1 > 0:
               illegal_move = False
            elif rnum >= 0.75 and xs[i-1] + 1 < arena_x:
               illegal_move = False

         # Fly moves up
         if rnum < 0.25:
            xs[i] = xs[i-1]
            ys[i] = ys[i-1] + 1
         #Fly moves down
         elif rnum >= 0.25 and rnum < 0.5:
            xs[i] = xs[i-1]
            ys[i] = ys[i-1] - 1
         # Fly moves left
         elif rnum >= 0.5 and rnum < 0.75:
            xs[i] = xs[i-1] - 1
            ys[i] = ys[i-1]
         # Fly moves right
         else:
            xs[i] = xs[i-1] + 1
            ys[i] = ys[i-1]
         
         # Now testing if current position is within the odor location
         if (xs[i] - starting_position_x)**2 + (ys[i] - starting_position_y)**2  <= odor_radius **2:
            ss1_num_successes += 1
            break
      ss1_num_completed_trials += 1

   plt.figure(figsize=(15, 3))
   plt.pcolormesh(arena_odor, alpha=0.5)
   plt.xlabel('X Direction')
   plt.ylabel('Y Direction')
   plt.title('Example of a Random Walk Navigator with Search Strategy 1')
   plt.colorbar()
   plt.plot(xs+0.5, ys+0.5, color='black')
   if save:
      plt.savefig('./images/odor_env_ss1.png', dpi=100, bbox_inches='tight')
   plt.show()

   return ss1_num_successes, ss1_num_completed_trials

def strat2(arena_odor, starting_position_x, starting_position_y, odor_radius, fly_sensitivity = 0.10, refresh_rate = 0.10, arena_x=100, arena_y=20, num_trials=10000, fly_time=2500, save=False):
   # Want to generate a loop for the fly to interact with the environment
   brw = False
   # Arena distribution 
   arena_fly_start_y = arena_y / 2
   arena_fly_start_x = arena_x - 1
   ss2_num_successes = 0
   ss2_num_completed_trials = 0

   # Search strategy 2
   for j in tqdm(range(num_trials)):
      xs = np.zeros(fly_time)
      ys = np.zeros(fly_time)

      xs[0] = arena_fly_start_x
      ys[0] = arena_fly_start_y

      # Now generating the random walk
      for i in range(1, fly_time):
         # Testing to see if the 
         if nr.rand() < fly_sensitivity and brw == False:
            brw = True
            beta = arena_odor[int(ys[i-1]), int(xs[i-1])]
         elif nr.rand() < refresh_rate and brw == True:
            brw = False
         
         illegal_move = True

         # Generating the probability
         if brw == False:
            while illegal_move:
               rnum = nr.rand()
               # Testing to see if an illegal move and the 
               # random num needs to be generated again
               if rnum < 0.25 and ys[i-1] + 1 < arena_y: 
                  illegal_move = False
               elif rnum >= 0.25 and rnum < 0.5 and ys[i-1] - 1 > 0:
                  illegal_move = False
               elif rnum >= 0.5 and rnum < 0.75 and xs[i-1] - 1 > 0:
                  illegal_move = False
               elif rnum >= 0.75 and xs[i-1] + 1 < arena_x:
                  illegal_move = False
            # Fly moves up
            if rnum < 0.25:
               xs[i] = xs[i-1]
               ys[i] = ys[i-1] + 1
            #Fly moves down
            elif rnum >= 0.25 and rnum < 0.5:
               xs[i] = xs[i-1]
               ys[i] = ys[i-1] - 1
            # Fly moves left
            elif rnum >= 0.5 and rnum < 0.75:
               xs[i] = xs[i-1] - 1
               ys[i] = ys[i-1]
            # Fly moves right
            else:
               xs[i] = xs[i-1] + 1
               ys[i] = ys[i-1]
         else:
            while illegal_move:
               rnum = nr.rand()
               # Testing to see if an illegal move and the 
               # random num needs to be generated again
               if rnum < 0.25 - (0.25*beta) and ys[i-1] + 1 < arena_y: 
                  illegal_move = False
               elif rnum >= 0.25 - (0.25*beta) and rnum < 2 * (0.25 - (0.25*beta)) and ys[i-1] - 1 > 0:
                  illegal_move = False
               elif rnum >= 2 * (0.25 - (0.25*beta)) and rnum < 3 * (0.25 - (0.25*beta)) and xs[i-1] + 1 < arena_x:
                  illegal_move = False
               elif rnum >= 3 * (0.25 - (0.25*beta)) and xs[i-1] - 1 > 0:
                  illegal_move = False
            
            # Now performing the biased random walk
            # Fly moves up
            if rnum < 0.25 - (0.25*beta):
               xs[i] = xs[i-1]
               ys[i] = ys[i-1] + 1
            #Fly moves down
            elif rnum >= 0.25 - (0.25*beta) and rnum < 2 * (0.25 - (0.25*beta)):
               xs[i] = xs[i-1]
               ys[i] = ys[i-1] - 1
            # Fly moves left
            elif rnum >= 2 * (0.25 - (0.25*beta)) and rnum < 3 * (0.25 - (0.25*beta)):
               xs[i] = xs[i-1] + 1
               ys[i] = ys[i-1]
            # Fly moves left
            else:
               xs[i] = xs[i-1] - 1
               ys[i] = ys[i-1]

         # Now testing if current position is within the odor location
         if (xs[i] - starting_position_x)**2 + (ys[i] - starting_position_y)**2  <= odor_radius **2:
            ss2_num_successes += 1
            xs = xs[:i]
            ys = ys[:i]
            break

      ss2_num_completed_trials += 1
   
   plt.figure(figsize=(15, 3))
   plt.pcolormesh(arena_odor, alpha=0.5)
   plt.xlabel('X Direction')
   plt.ylabel('Y Direction')
   plt.title('Example of a Random Walk Navigator with Search Strategy 2')
   plt.colorbar()
   plt.plot(xs+0.5, ys+0.5, color='black')
   if save:
      plt.savefig('./images/odor_env_ss2.png', dpi=100, bbox_inches='tight')
   plt.show()
   # Returning the final thing
   return ss2_num_successes, ss2_num_completed_trials

def strat3(arena_odor, starting_position_x, starting_position_y, odor_radius, fly_sensitivity = 0.10, refresh_rate = 0.10, arena_x=100, arena_y=20, num_trials=10000, fly_time=2500, save=False):
   # Want to generate a loop for the fly to interact with the environment
   brw = False
   arena_fly_start_y = arena_y / 2
   arena_fly_start_x = arena_x - 1
   ss3_num_successes = 0
   ss3_num_completed_trials = 0

   # Search strategy 3
   for j in tqdm(range(num_trials)):
      xs = np.zeros(fly_time)
      ys = np.zeros(fly_time)

      xs[0] = arena_fly_start_x
      ys[0] = arena_fly_start_y

      # Now generating the random walk
      for i in range(1, fly_time):
         # Testing to see if the 
         if nr.rand() < fly_sensitivity and brw == False:
            brw = True
            beta = arena_odor[int(ys[i-1]), int(xs[i-1])]
            # Choosing direction in which to bias up or down 
            if ys[i-1] + 1 > arena_y:
               down = True
            elif ys[i-1] - 1 > 0:
               down = False
            else:
               if arena_odor[int(ys[i-1] + 1), int(xs[i-1])] > arena_odor[int(ys[i-1] - 1), int(xs[i-1])]:
                  down = False
               else:
                  down = True
         elif nr.rand() < refresh_rate and brw == True:
            brw = False
         
         illegal_move = True
         # Generating the probability
         if brw == False:
            while illegal_move:
               rnum = nr.rand()
               # Testing to see if an illegal move and the 
               # random num needs to be generated again
               if rnum < 0.25 and ys[i-1] + 1 < arena_y: 
                  illegal_move = False
               elif rnum >= 0.25 and rnum < 0.5 and ys[i-1] - 1 > 0:
                  illegal_move = False
               elif rnum >= 0.5 and rnum < 0.75 and xs[i-1] - 1 > 0:
                  illegal_move = False
               elif rnum >= 0.75 and xs[i-1] + 1 < arena_x:
                  illegal_move = False
            # Fly moves up
            if rnum < 0.25:
               xs[i] = xs[i-1]
               ys[i] = ys[i-1] + 1
            #Fly moves down
            elif rnum >= 0.25 and rnum < 0.5:
               xs[i] = xs[i-1]
               ys[i] = ys[i-1] - 1
            # Fly moves left
            elif rnum >= 0.5 and rnum < 0.75:
               xs[i] = xs[i-1] - 1
               ys[i] = ys[i-1]
            # Fly moves right
            else:
               xs[i] = xs[i-1] + 1
               ys[i] = ys[i-1]
         else:
            if down:
               while illegal_move:
                  rnum = nr.rand()
                  # Testing to see if an illegal move and the 
                  # random num needs to be generated again
                  if rnum < 0.25 + (0.25*beta) and ys[i-1] - 1 > 0: 
                     illegal_move = False
                  elif rnum >= 0.25 + (0.25*beta) and rnum < (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)) and ys[i-1] + 1 < arena_y:
                     illegal_move = False
                  elif rnum >= (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)) and rnum < (0.25 + (0.25*beta)) + 2 *(0.25 - (0.25*beta)) and xs[i-1] + 1 < arena_x:
                     illegal_move = False
                  elif rnum >= (0.25 + (0.25*beta)) + 2 *(0.25 - (0.25*beta)) and xs[i-1] - 1 > 0:
                     illegal_move = False
               
               # Now performing the biased random walk
               # Fly moves up
               if rnum < 0.25 + (0.25*beta):
                  xs[i] = xs[i-1]
                  ys[i] = ys[i-1] - 1
               #Fly moves down
               elif rnum >= 0.25 + (0.25*beta) and rnum < (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)):
                  xs[i] = xs[i-1]
                  ys[i] = ys[i-1] + 1
               # Fly moves left
               elif rnum >= (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)) and rnum < (0.25 + (0.25*beta)) + 2 *(0.25 - (0.25*beta)):
                  xs[i] = xs[i-1] + 1
                  ys[i] = ys[i-1]
               # Fly moves left
               else:
                  xs[i] = xs[i-1] - 1
                  ys[i] = ys[i-1]

            else:
               while illegal_move:
                  rnum = nr.rand()
                  # Testing to see if an illegal move and the 
                  # random num needs to be generated again
                  if rnum < 0.25 + (0.25*beta) and ys[i-1] + 1 < arena_y: 
                     illegal_move = False
                  elif rnum >= 0.25 + (0.25*beta) and rnum < (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)) and ys[i-1] - 1 > 0:
                     illegal_move = False
                  elif rnum >= (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)) and rnum < (0.25 + (0.25*beta)) + 2 *(0.25 - (0.25*beta)) and xs[i-1] + 1 < arena_x:
                     illegal_move = False
                  elif rnum >= (0.25 + (0.25*beta)) + 2 *(0.25 - (0.25*beta)) and xs[i-1] - 1 > 0:
                     illegal_move = False
               
               # Now performing the biased random walk
               # Fly moves up
               if rnum < 0.25 + (0.25*beta):
                  xs[i] = xs[i-1]
                  ys[i] = ys[i-1] + 1
               #Fly moves down
               elif rnum >= 0.25 + (0.25*beta) and rnum < (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)):
                  xs[i] = xs[i-1]
                  ys[i] = ys[i-1] - 1
               # Fly moves left
               elif rnum >= (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)) and rnum < (0.25 + (0.25*beta)) + 2 *(0.25 - (0.25*beta)):
                  xs[i] = xs[i-1] + 1
                  ys[i] = ys[i-1]
               # Fly moves left
               else:
                  xs[i] = xs[i-1] - 1
                  ys[i] = ys[i-1]
         # Now testing if current position is within the odor location
         if (xs[i] - starting_position_x)**2 + (ys[i] - starting_position_y)**2  <= odor_radius **2:
            ss3_num_successes += 1
            xs = xs[:i]
            ys = ys[:i]
            break

      ss3_num_completed_trials += 1
         
   plt.figure(figsize=(15, 3))
   plt.pcolormesh(arena_odor, alpha=0.5)
   plt.xlabel('X Direction')
   plt.ylabel('Y Direction')
   plt.title('Example of a Random Walk Navigator with Search Strategy 3')
   plt.colorbar()
   plt.plot(xs+0.5, ys+0.5, color='black')
   if save:
      plt.savefig('./images/odor_env_ss3.png', dpi=100, bbox_inches='tight')
   plt.show()

   return ss3_num_successes, ss3_num_completed_trials

def strat4(arena_odor, starting_position_x, starting_position_y, odor_radius, fly_sensitivity = 0.10, refresh_rate = 0.10, arena_x=100, arena_y=20, num_trials=10000, fly_time=2500, save=False):
   # Want to generate a loop for the fly to interact with the environment
   brw = False
   arena_fly_start_y = arena_y / 2
   arena_fly_start_x = arena_x - 1
   ss4_num_successes = 0
   ss4_num_completed_trials = 0

   # Search strategy 3
   for j in tqdm(range(num_trials)):
      xs = np.zeros(fly_time)
      ys = np.zeros(fly_time)

      xs[0] = arena_fly_start_x
      ys[0] = arena_fly_start_y

      # Now generating the random walk
      for i in range(1, fly_time):
         # Testing to see if the 
         if nr.rand() < fly_sensitivity and brw == False:
            brw = True
            # Choosing direction in which to bias up or down 
            if ys[i-1] + 1 > arena_y:
               down = True
            elif ys[i-1] - 1 > 0:
               down = False
            else:
               if arena_odor[int(ys[i-1] + 1), int(xs[i-1])] > arena_odor[int(ys[i-1] - 1), int(xs[i-1])]:
                  down = False
               else:
                  down = True
         elif nr.rand() < refresh_rate and brw == True:
            brw = False
         
         illegal_move = True
         # Generating the probability
         if brw == False:
            while illegal_move:
               rnum = nr.rand()
               # Testing to see if an illegal move and the 
               # random num needs to be generated again
               if rnum < 0.25 and ys[i-1] + 1 < arena_y: 
                  illegal_move = False
               elif rnum >= 0.25 and rnum < 0.5 and ys[i-1] - 1 > 0:
                  illegal_move = False
               elif rnum >= 0.5 and rnum < 0.75 and xs[i-1] - 1 > 0:
                  illegal_move = False
               elif rnum >= 0.75 and xs[i-1] + 1 < arena_x:
                  illegal_move = False
            # Fly moves up
            if rnum < 0.25:
               xs[i] = xs[i-1]
               ys[i] = ys[i-1] + 1
            #Fly moves down
            elif rnum >= 0.25 and rnum < 0.5:
               xs[i] = xs[i-1]
               ys[i] = ys[i-1] - 1
            # Fly moves left
            elif rnum >= 0.5 and rnum < 0.75:
               xs[i] = xs[i-1] - 1
               ys[i] = ys[i-1]
            # Fly moves right
            else:
               xs[i] = xs[i-1] + 1
               ys[i] = ys[i-1]
         else:
            beta = arena_odor[int(ys[i-1]), int(xs[i-1])]
            if down:
               while illegal_move:
                  rnum = nr.rand()
                  # Testing to see if an illegal move and the 
                  # random num needs to be generated again
                  if rnum < 0.25 + (0.25*beta) and ys[i-1] - 1 > 0: 
                     illegal_move = False
                  elif rnum >= 0.25 + (0.25*beta) and rnum < (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)) and ys[i-1] + 1 < arena_y:
                     illegal_move = False
                  elif rnum >= (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)) and rnum < (0.25 + (0.25*beta)) + 2 *(0.25 - (0.25*beta)) and xs[i-1] + 1 < arena_x:
                     illegal_move = False
                  elif rnum >= (0.25 + (0.25*beta)) + 2 *(0.25 - (0.25*beta)) and xs[i-1] - 1 > 0:
                     illegal_move = False
               
               # Now performing the biased random walk
               # Fly moves up
               if rnum < 0.25 + (0.25*beta):
                  xs[i] = xs[i-1]
                  ys[i] = ys[i-1] - 1
               #Fly moves down
               elif rnum >= 0.25 + (0.25*beta) and rnum < (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)):
                  xs[i] = xs[i-1]
                  ys[i] = ys[i-1] + 1
               # Fly moves left
               elif rnum >= (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)) and rnum < (0.25 + (0.25*beta)) + 2 *(0.25 - (0.25*beta)):
                  xs[i] = xs[i-1] + 1
                  ys[i] = ys[i-1]
               # Fly moves left
               else:
                  xs[i] = xs[i-1] - 1
                  ys[i] = ys[i-1]
            else:
               while illegal_move:
                  rnum = nr.rand()
                  # Testing to see if an illegal move and the 
                  # random num needs to be generated again
                  if rnum < 0.25 + (0.25*beta) and ys[i-1] + 1 < arena_y: 
                     illegal_move = False
                  elif rnum >= 0.25 + (0.25*beta) and rnum < (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)) and ys[i-1] - 1 > 0:
                     illegal_move = False
                  elif rnum >= (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)) and rnum < (0.25 + (0.25*beta)) + 2 *(0.25 - (0.25*beta)) and xs[i-1] + 1 < arena_x:
                     illegal_move = False
                  elif rnum >= (0.25 + (0.25*beta)) + 2 *(0.25 - (0.25*beta)) and xs[i-1] - 1 > 0:
                     illegal_move = False
               
               # Now performing the biased random walk
               # Fly moves up
               if rnum < 0.25 + (0.25*beta):
                  xs[i] = xs[i-1]
                  ys[i] = ys[i-1] + 1
               #Fly moves down
               elif rnum >= 0.25 + (0.25*beta) and rnum < (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)):
                  xs[i] = xs[i-1]
                  ys[i] = ys[i-1] - 1
               # Fly moves left
               elif rnum >= (0.25 + (0.25*beta)) + (0.25 - (0.25*beta)) and rnum < (0.25 + (0.25*beta)) + 2 *(0.25 - (0.25*beta)):
                  xs[i] = xs[i-1] + 1
                  ys[i] = ys[i-1]
               # Fly moves left
               else:
                  xs[i] = xs[i-1] - 1
                  ys[i] = ys[i-1]
         # Now testing if current position is within the odor location
         if (xs[i] - starting_position_x)**2 + (ys[i] - starting_position_y)**2  <= odor_radius **2:
            ss4_num_successes += 1
            xs = xs[:i]
            ys = ys[:i]
            break

      ss4_num_completed_trials += 1
         
   plt.figure(figsize=(15, 3))
   plt.pcolormesh(arena_odor, alpha=0.5)
   plt.xlabel('X Direction')
   plt.ylabel('Y Direction')
   plt.title('Example of a Random Walk Navigator with Search Strategy 4')
   plt.colorbar()
   plt.plot(xs+0.5, ys+0.5, color='black')
   if save:
      plt.savefig('./images/odor_env_ss4png', dpi=100, bbox_inches='tight')
   plt.show()
   return ss4_num_successes, ss4_num_completed_trials