# -*- coding: utf-8 -*-
# Importiere die nötigen Packete
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import time
import matplotlib
import random
from contextlib import suppress
random.seed(4) # wichtig
np.random.seed(seed=42)

def print_world(world):
  """Printe die eingegebene Matrix farbig"""
  plt.figure(figsize=(20,10))
  bounds = [0,1,10,50,100,300]
  colors = ["green","red","red","white","yellow", "black"]
  cmap = matplotlib.colors.ListedColormap(colors)
  norm = matplotlib.colors.BoundaryNorm(bounds, (len(colors)-1))
  plt.imshow(world, cmap=cmap, norm= norm)
  plt.show()


def shifting(world, x1, axis1, x2, axis2):
    """Shifte Matrix in alle möglichen Richtungen"""
    return np.roll(np.roll(world, x2, axis2), x1, axis1)

def start_points(n, world):
    """Setze Ecken und Mitte der eingegebenen Matrix auf 1"""
    world[0, 0] = 1
    world[n-1, n-1] = 1
    world[0, n-1] = 1
    world[n-1, 0] = 1
    world[np.round(n/2).astype(int)][np.round(n/2).astype(int)] = 1
    return world
    
def build_world(n, threshold):
    """Baue die Welt in der eingegeben Größe, mit 'Stadtkernen' in der Mitte und den Rändern"""
    #Berechne die zu füllenden Zellen
    percentage = lambda x, n: np.sum(np.where(x >1, 1, x)) / (n*n) * 100
    world = np.zeros(n*n).reshape(n, n)
    world = start_points(n=n, world=world)
    #Solange es in der Welt nicht ausreichend gefüllte Zellen gibt, fülle...
    worldnotfull  = True
    while worldnotfull:  
        if percentage(world, n) <= threshold:
            world += shifting(world, 1, 1, 0, 0) #rechts
        else:
            worldnotfull = False
            break
        if percentage(world, n) <= threshold:
            world += shifting(world, 1, 0, 0, 0) #unten
        else: 
            worldnotfull = False
            break
            
        if percentage(world, n) <= threshold:
            world += shifting(world, -1, 1, 0, 1) #links
        else:
            worldnotfull = False
            break
            
        if percentage(world, n) <= threshold:
            world += shifting(world, -1, 1, 0, 0) #oben
        else:
            worldnotfull = False
            break

        if percentage(world, n) <= threshold:
            world += shifting(world, 1, 1, 1, 0) #rechts unten
        else:
            worldnotfull = False
            break
        if percentage(world, n) <= threshold:
            world += shifting(world, -1, 0, -1, 1) #links oben
        else:
            worldnotfull = False
            break
        if percentage(world, n) <= threshold:
            world += shifting(world, -1, 0, 1, 1) #rechts oben
        else:
            worldnotfull = False
            break
            
        if percentage(world, n) <= threshold:
            world += shifting(world, -1, 1, 1, 0) #links unten
        else:
            worldnotfull = False
            break
    #Setze zu große Werte zurück und gebe Welt aus
    world = np.where(world >1, 0, 50)
    #Checke ob min. eine Zelle infiziert ist, wenn nicht, führe aus bis min. eine infiziert
    worldinfected = True
    while worldinfected == True:
      for i, v in np.ndenumerate(world):
        if v == 0:
          world[i] = np.random.choice([0, 1], p=[0.99, 0.01])
        got_infections = np.any(world == 1)
        if got_infections == True:
          worldinfected = False
    return world


def movement_property(n):
  """Gebe Zellen eine Zahl, die die Bewegungseigenschaften beschreibt"""
  radius_world = np.zeros(n*n).reshape(n, n)
  for i, v in np.ndenumerate(radius_world):
      radius_world[i] = np.random.choice([1, 2, 4, 6], p=[0.85, 0.1, 0.025, 0.025])
  return radius_world

def infection_probability(mu, sigma, n=1):
    """Ziehe wahrscheinlichkeit aus Normalverteilung"""
    prob = np.abs(np.random.normal(mu, sigma, n))
    
    if (prob + prob) <0 or (prob + prob) >1: 
        prob = np.array([mu])
    prob = prob.tolist()[0]
    return prob

def infection(init_world, neighbours_world,next_world, x, y):
  """Ziehe einen zufälligen Wert und infiziere ggf. Zelle"""
  if init_world[x][y] == 0:
    if 1 <= neighbours_world[x,y] <= 4:
      prob = infection_probability(mu=0.4, sigma=0.1)
      poss_infection = np.random.choice([0,1], p=[1-prob, prob])
      next_world[x][y] = poss_infection
      
    if 5 <= neighbours_world[x,y] <= 8:
      prob = infection_probability(mu=0.5, sigma=0.1)
      poss_infection = np.random.choice([0,1], p=[1-prob, prob])
      next_world[x][y] = poss_infection

    if 9 <= neighbours_world[x,y] <= 10:
      prob = infection_probability(mu=0.7, sigma=0.1)
      poss_infection = np.random.choice([0,1], p=[1-prob, prob])
      next_world[x][y] = poss_infection

    if neighbours_world[x,y] >= 11:
      prob = infection_probability(mu=0.9, sigma=0.1)
      poss_infection = np.random.choice([0,1], p=[1-prob, prob])
      next_world[x][y] = poss_infection
  return next_world

def tracker(world): 
    """Setze Zellen nach 8 Epochen auf immun, markiere infizierte Zellen als True"""
    world[world==8] +=92 
    
    conditions = [(world >=1) & (world <10), world >= 50] #Quelle: https://stackoverflow.com/questions/39109045/numpy-where-with-multiple-conditions/39111919
    choices = [True, False]
    infected = np.select(conditions, choices, default=False) #default = False, bedeutet, dass alle, die nicht davon erfasst sind False sind
    return infected

def count_neighbours(world, radius, x, y, n):
  """Zähle die Nachbarzellen und gebe sie aus."""

  neighbours = 0
  current_position = (x,y)
  for a in range(x-radius if x-radius >= 0 else 0 ,x+radius+1 if x+radius+1 <= n else n):
    for b in range(y-radius if y-radius >= 0 else 0 ,y+radius+1 if y+radius+1 <= n else n):
      if (a,b) != current_position:
        if  1 <= world[a,b] < 10:
          neighbours += 1
  return neighbours
    

def die(world, percentage):
    """Ziehe Stichprobe aus den infizierten Zellen und stelle Wert auf 300 ('Tot')"""

    
    infected = np.sum((world >= 1) & (world <= 10))
    to_die = percentage * infected
    if to_die < 1:
      to_die = 0
    else:
      to_die = to_die
    to_die = np.round(to_die).astype(int)


    indizes = [] # Für die Koordinaten der infizierten Zellen
    for i, v in np.ndenumerate(world):
        if v in range(1, 11):
          indizes.append(i)
    #Ziehe Stichprobe aus den infizierten Zellen und setze sie auf 300
    sample = random.sample(indizes, to_die)
    for i in sample:
        world[i] = 300
  
    return world
    
def next_period(world, epoche, n, die_percentage, lockdown, lockdown_request, population):
  """Hauptschleife; Zähle Epochen und gehe durch die einzelnen Zellen
     zähle die Nachbarn, infiziere, immunisiere und lasse Zellen sterben; 
     printe die Welt nach jeder Iteration
     """
  # Initialisierung
  die_percentage = die_percentage
  population = population   
  lockdown = lockdown
  lockdown_request = lockdown_request
  init_world = world
  print_world(init_world)
  epoche = epoche
  count = 1
  # Für die plots
  list_dead = []
  list_immune = []
  list_infected = []
  list_healthy = []
  list_healthy.append(np.sum(init_world == 0))
  list_dead.append(np.sum(init_world == 300))
  list_immune.append(np.sum(init_world == 100))
  list_infected.append(np.sum((init_world > 0) & (init_world < 10)))
  plot_epochs = epoche

  # Laufe, bis Anzahl der eingegebenen Epochen erreicht
  while count != epoche:
    next_world = deepcopy(init_world)
    neighbours_world = np.zeros((n,n))
    infected = tracker(next_world)
    next_world = infected + next_world
    next_world = die(next_world, die_percentage)
    # Für die plots
    list_healthy.append(np.sum(next_world == 0))
    list_dead.append(np.sum(next_world == 300))
    list_immune.append(np.sum(next_world == 100))
    list_infected.append(np.sum((next_world > 0) & (next_world < 10)))
    if count % lockdown_request == 0:
      try: 
        lockdown_input = str(input("Lockdown yes or no: "))
      except ValueError:
        print("Sorry, I didn't understand that.") 
        continue
      if lockdown_input == "yes":
        lockdown = True
      elif lockdown_input == "no":
        lockdown = False
      else:
        print("Please enter yes or no.")
    if lockdown == False:
      radius_world = movement_property(n)
      # Zähle Nachbarn und infiziere Zellen
      for x in range (0,n):
        for y in range (0,n): 
          if radius_world[x][y] == 1:
            radius = 1
            neighbours_world[x,y] = count_neighbours(init_world, radius, x, y, n)
            next_world = infection(init_world, neighbours_world, next_world, x, y)

          if radius_world[x][y] == 2:
            radius = 2
            neighbours_world[x,y]  = count_neighbours(init_world, radius, x, y, n)
            next_world = infection(init_world, neighbours_world, next_world, x, y)
            
          if radius_world[x][y] == 4:
            radius = 4
            neighbours_world[x,y]  = count_neighbours(init_world, radius, x, y, n)
            next_world = infection(init_world, neighbours_world, next_world, x, y)
            
          if radius_world[x][y] == 6:
            radius = 6
            neighbours_world[x,y]  = count_neighbours(init_world, radius, x, y, n)
            next_world = infection(init_world, neighbours_world, next_world, x, y)
          
            
  
      count += 1

      # print world und inforInformationmation
      infected = np.sum((next_world >= 1) & (next_world <= 10))
      healthy = np.sum(next_world ==0)
      dead = np.sum(next_world==300)
      immune = np.sum(next_world==100)
      print(f'Epoche: {count}/{epoche}, healthy: {healthy}/{population}, infected: {infected}/{population}, dead: {dead}/{population}, immune: {immune}/{population}')
      print_world(next_world)
      del(init_world)
      init_world = deepcopy(next_world)

    # kein lockdown
    else:
      for x in range (0,n):
        for y in range (0,n): 
            radius = 1
            neighbours_world[x,y]  = count_neighbours(init_world,radius, x, y, n)
            next_world = infection(init_world, neighbours_world, next_world, x, y)
          
      
      count += 1

      # print world und Information
      infected = np.sum((next_world >= 1) & (next_world <= 10))
      healthy = np.sum(next_world ==0)
      dead = np.sum(next_world==300)
      immune = np.sum(next_world==100)
      print(f'Epoche: {count}/{epoche}, healthy: {healthy}/{population}, infected: {infected}/{population}, dead: {dead}/{population}, immune: {immune}/{population}')
      print_world(next_world)

      del(init_world)
      init_world = deepcopy(next_world)
  # Am Ende: plote die oben gesammelten Daten aus der Liste
  plt.plot(list_healthy, color="green")
  plt.plot(list_immune, color="yellow")
  plt.plot(list_infected, color="red")
  plt.plot(list_dead, color="black")
  plt.legend(("healthy", "immune", "infected", "dead"),handlelength=1.5, fontsize=16)
  plt.title("Plot of the development")
  plt.xlabel("Epoch")
  plt.ylabel("Number of cells")
  plt.xlim(0, len(list_healthy)-1)

def start_pandemic():
    """Füge alle Funktionen Zusammen, stelle Fragen nach den Parametern 
       und starte den Automaten
       """
    print("Virus spread simulation")
    question1 = True
    question2 = True
    question3 = True
    question4 = True
    question5 = True
    question6 = True
    epoche=None
    lockdown = False
    lockdown_request = None

    # Stelle Fragen nach Parametern
    while question1:
      try:
        n = int(input("How large should the game field be ? : "))
      except ValueError:
        print("Sorry, I didn't understand that.")
        continue
      if n < 0:
        print("Sorry, your response must not be negative.")
        continue
      else:
        break

    while question2:
      try:   
        threshold = float(input("What percentage of the field should be populated? ? Please enter a number between 0 and 100. "))
      except ValueError:
        print("Sorry, I didn't understand that.")
        continue
      if threshold <= 0 or threshold > 100:
        print('Please enter number in range.')
        continue
      else:
        break

    while question3:
        try:
          epoche = int(input("Enter a number of epochs: "))
        except ValueError:
          print("Sorry, I didn't understand that.")
          continue
        if epoche <0: #falls negativ, damit kein unendlicher loop entsteht
          print("Sorry, your response must not be negative.")
          continue
        else:
          break
      

    while question4:
      try: 
        lockdown_input = str(input("Lockdown yes or no: "))
      except ValueError:
        print("Sorry, I didn't understand that.") 
        continue
      if lockdown_input == "yes":
        lockdown = True
        break
      elif lockdown_input == "no":
        lockdown = False
        break
      else:
        print("Please enter yes or no.")
        continue

    while question5:
      try: 
        lockdown_request = int(input("After how many epoches do you want to decide to activate lockdown: "))
      except ValueError:
        print("Sorry, I didn't understand that.") 
        continue
      if 0 <=  lockdown_request > epoche:
        print(f"Please enter a number between 0 and {epoche}.") 
        continue
      else:
        break 

    while question6:
      try: 
        die_percentage = float(input("How many cells shall die each round, in percent?"))
      except ValueError:
        print("Sorry, I didn't understand that.")
        continue
      if die_percentage < 0 or die_percentage > 1:
        print('Please enter number between 0 and 1.')
        continue
      else:
        break
    # Starte Automaten
    world = build_world(n, threshold)
    

    init_infected = np.sum(world ==1)
    init_healthy = np.sum(world ==0)
    population = init_infected + init_healthy
    print(f'Epoche: {1}/{epoche}, healthy: {init_healthy}/{population}, infected: {init_infected}/{population}, dead: {0}/{population}, immune: {0}/{population}')
    next_period(world, epoche, n, die_percentage, lockdown, lockdown_request, population)

start_pandemic()
