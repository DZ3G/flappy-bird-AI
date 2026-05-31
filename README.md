# Flappy Bird AI

I built an AI that teaches itself to play Flappy Bird. It starts out terrible — the birds fly straight into the first pipe — and after about 10 generations one of them figures it out and just... never dies. It's genuinely cool to watch.

---

## Table of Contents

1. [What this is](#1-what-this-is)
2. [How it actually works](#2-how-it-actually-works)
3. [File structure](#3-file-structure)
4. [Setup](#4-setup)
5. [Main.py explained](#5-mainpy-explained)
6. [The config file](#6-the-config-file)
7. [How to run it](#7-how-to-run-it)
8. [What you'll see](#8-what-youll-see)
9. [NEAT explained simply](#9-neat-explained-simply)
10. [Common errors](#10-common-errors)

---

## 1. What this is

50 birds play Flappy Bird at the same time. Each bird has its own tiny neural network (basically a brain). The ones that survive longest pass their "genes" to the next generation, with small random mutations. After around 10 generations, one bird figures out the perfect timing and just goes forever.

The algorithm behind this is called **NEAT** (NeuroEvolution of Augmenting Topologies). It was invented by researchers at UT Austin in 2002 and it's basically Darwinian evolution for neural networks.

---

## 2. How it actually works

```
Gen 1:  50 birds spawn → all crash immediately (random brains)
                ↓
        Best performers kept
                ↓
Gen 2:  50 new birds (slightly better brains)
                ↓
            repeat...
                ↓
Gen ~10: one bird clears every pipe forever 🏆
```

### What each bird "sees" (the 3 inputs)

Each bird's brain gets 3 numbers every frame:

```
1. Bird's current height (bird.y)
2. Distance from bird to the TOP of the next pipe
3. Distance from bird to the BOTTOM of the next pipe
```

Based on those 3 numbers, the brain outputs one number. If it's above 0.5 → jump. Below 0.5 → do nothing.

### How fitness (score) is calculated

| Event | Points |
|-------|--------|
| Surviving each frame | +0.1 |
| Clearing a pipe | +5.0 |
| Hitting something | -1.0 |

---

## 3. File structure

```
flappy-bird-AI/
│
├── Main.py                  ← the actual game + AI code
├── config-feedforward.txt   ← settings for the NEAT algorithm
├── README.md                ← you're reading it
│
└── IMGs/
    ├── Bird.png
    ├── pipe.png
    ├── bg.png
    └── base.png
```

> The `IMGs` folder has to be in the same folder as `Main.py` or the game crashes immediately on startup.

---

## 4. Setup

### Check Python is installed

```bash
python --version
```

You need Python 3. If that command doesn't work, download it from [python.org](https://python.org).

### Install the two libraries

```bash
pip install pygame neat-python
```

| Library | What it does |
|---------|-------------|
| `pygame` | Renders the game window, handles images and collision |
| `neat-python` | Runs the NEAT algorithm, evolves the bird brains |

---

## 5. Main.py explained

### Imports and constants

```python
import pygame      # game window and graphics
import neat        # the AI algorithm
import os          # for finding image files on disk
import random      # random pipe heights

WIN_W = 500        # window width in pixels
WIN_H = 800        # window height
FPS   = 30         # frames per second
```

### Loading images

```python
IMG_PATH = os.path.join(os.path.dirname(__file__), "imgs")
# finds the imgs folder wherever the script is located

BIRD_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join(IMG_PATH, "bird.png")))
# scale2x doubles the image size, otherwise the sprites are tiny
```

### The Bird class

Handles all the physics for one bird.

```python
class Bird:
    MAX_VEL = 10    # max falling speed
    GRAVITY  = 1.5  # added to velocity every frame

    def __init__(self, x, y):
        self.x   = x
        self.y   = float(y)
        self.vel = 0        # starts stationary
```

`jump()` — called when the AI decides to jump:
```python
def jump(self):
    self.vel = -10.5
    # negative because in pygame y=0 is the TOP of the screen
    # negative velocity = moving upward
```

`move()` — called every frame to apply gravity:
```python
def move(self):
    self.vel = min(self.vel + self.GRAVITY, self.MAX_VEL)
    self.y  += self.vel
```

`get_mask()` — for pixel-perfect collision:
```python
def get_mask(self):
    return pygame.mask.from_surface(BIRD_IMG)
    # checks the actual shape of the bird, not just a rectangle
```

### The Pipe class

```python
class Pipe:
    GAP = 200   # space between top and bottom pipe
    VEL =   5   # how fast pipes move left

    def __init__(self, x):
        self.height = random.randrange(80, 450)
        self.top    = self.height - PIPE_IMG.get_height()
        self.bottom = self.height + self.GAP
        self.passed = False

        # top pipe is just the image flipped upside down
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
```

`collide()` — pixel-level collision detection:
```python
def collide(self, bird):
    bird_mask = bird.get_mask()
    top_mask  = pygame.mask.from_surface(self.PIPE_TOP)
    bot_mask  = pygame.mask.from_surface(PIPE_IMG)

    top_offset = (self.x - int(bird.x), self.top    - round(bird.y))
    bot_offset = (self.x - int(bird.x), self.bottom - round(bird.y))

    return (bird_mask.overlap(top_mask, top_offset) or
            bird_mask.overlap(bot_mask, bot_offset))
```

### draw_window()

Redraws everything each frame in the right order:

```python
def draw_window(win, birds, pipes, score, generation):
    win.blit(BG_IMG, (0, 0))   # 1. background (bottom layer)
    for pipe in pipes:
        pipe.draw(win)          # 2. pipes
    for bird in birds:
        bird.draw(win)          # 3. birds (on top of pipes)
    # 4. score text on top of everything
    pygame.display.update()     # 5. push frame to screen
```

Order matters here — if you draw the background last it covers everything.

### eval_genomes() — the core AI function

NEAT calls this automatically at the start of each generation.

```python
def eval_genomes(genomes, config):
    global GEN
    GEN += 1

    # 3 lists kept in sync — same index = same bird
    nets  = []   # neural network for each bird
    birds = []   # Bird object for each bird
    ge    = []   # NEAT genome for each bird (used to update fitness)

    for _, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        ge.append(genome)
```

Decision to jump — runs every frame for every living bird:
```python
output = nets[i].activate((
    bird.y,
    abs(bird.y - pipes[pipe_idx].height),
    abs(bird.y - pipes[pipe_idx].bottom),
))

if output[0] > 0.5:
    bird.jump()
```

When a bird hits a pipe:
```python
if pipe.collide(birds[i]):
    ge[i].fitness -= 1
    birds.pop(i)
    nets.pop(i)
    ge.pop(i)
    # all 3 lists must be updated together or the indexes break
```

When a bird clears a pipe:
```python
if not pipe.passed and birds and birds[0].x > pipe.x + 60:
    pipe.passed = True
    score += 1
    for g in ge:
        g.fitness += 5       # reward every bird still alive
    pipes.append(Pipe(700))  # spawn next pipe
```

### run_neat() — starting the training

```python
def run_neat(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))

    winner = population.run(eval_genomes, 50)
    # runs up to 50 generations
    # stops early if a bird hits fitness_threshold = 100
```

---

## 6. The config file

`config-feedforward.txt` controls how NEAT behaves. The important parts:

```ini
[NEAT]
fitness_threshold       = 100   # stop when a bird reaches 100 pts
pop_size                = 50    # 50 birds per generation

[DefaultGenome]
num_inputs              = 3     # height, dist to top pipe, dist to bottom pipe
num_outputs             = 1     # jump or don't
num_hidden              = 0     # starts with no hidden layer (NEAT adds them if needed)

[DefaultReproduction]
elitism                 = 2     # top 2 always survive unchanged
survival_threshold      = 0.2   # only top 20% can reproduce

[DefaultStagnation]
max_stagnation          = 20    # kill a species if no improvement in 20 gens
```

> To make it learn faster, set `pop_size = 100`. More birds per generation = faster learning, but each generation takes slightly longer to simulate.

---

## 7. How to run it

### VS Code
Open the `flappy-bird-AI` folder, open `Main.py`, press **F5**.

### Terminal
```bash
cd path/to/flappy-bird-AI
python Main.py
```

---

## 8. What you'll see

The game window shows:
- `Score` — pipes cleared by the best bird this generation
- `Birds` — how many are still alive
- `Gen` — current generation number

Typical progression:

| Generation | What happens |
|------------|-------------|
| 1–2 | Everyone dies in the first few seconds |
| 3–5 | Some birds start jumping at roughly the right time |
| 6–10 | A bird clears 1 or 2 pipes |
| 10–15 | One bird goes forever |

The terminal also prints stats after each generation showing average fitness and the best score that round.

---

## 9. NEAT explained simply

NEAT stands for **NeuroEvolution of Augmenting Topologies**.

Normal machine learning trains one big neural network by adjusting weights. NEAT does something different — it evolves the structure of the network itself, adding neurons and connections over time, starting from almost nothing.

**The 3 key ideas:**

**Starting minimal** — each bird's brain starts with just 3 inputs wired directly to 1 output. NEAT only adds complexity when it helps. This is way faster than starting with a random large network.

**Speciation** — birds with similar brain structures are grouped into species and compete separately. This protects weird new mutations that need a few generations to prove themselves before getting wiped out by already-working brains.

**Innovation numbers** — when two brains reproduce, NEAT tracks which neurons correspond to each other using unique IDs. This solves the problem of combining two different brain structures without breaking them.

### Fitness formula in this project

```
fitness = (frames survived × 0.1) + (pipes cleared × 5.0) - (collisions × 1.0)

Example: survived 200 frames, cleared 3 pipes, hit 1 pipe
       = (200 × 0.1) + (3 × 5) - (1 × 1)
       = 20 + 15 - 1
       = 34.0 pts
```

---

## 10. Common errors

### `ModuleNotFoundError: No module named 'pygame'`
```bash
pip install pygame
```

### `ModuleNotFoundError: No module named 'neat'`
```bash
pip install neat-python
```
Note: the install name is `neat-python` but you import it as `import neat`.

### `Missing required configuration item: 'no_fitness_termination'`
Add this line to the `[NEAT]` section of your config file:
```ini
no_fitness_termination = False
```

### Black screen on startup
Check that the `IMGs` folder contains `Bird.png`, `pipe.png`, `bg.png`, and `base.png`. Filenames are case-sensitive on Mac/Linux — `Bird.png` and `bird.png` are different files.

### `python` not recognized in terminal
Try:
```bash
python3 Main.py
```

### VS Code using the wrong Python
`Ctrl+Shift+P` → **Python: Select Interpreter** → pick the one that has your packages installed.

---

## Sources

- Original NEAT paper — Stanley & Miikkulainen, 2002: [nn.cs.utexas.edu](https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf)
- neat-python docs: [neat-python.readthedocs.io](https://neat-python.readthedocs.io)
- Tutorial that inspired this: [Tech With Tim on YouTube](https://www.youtube.com/watch?v=OGHA-elMrxI)
- pygame docs: [pygame.org/docs](https://www.pygame.org/docs/)
