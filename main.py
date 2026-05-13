import pygame
import neat
import os
import random

# ─────────────────────────────────────────
#  WINDOW & GAME SETTINGS
# ─────────────────────────────────────────
WIN_W = 500
WIN_H = 800
FPS   = 30

# ─────────────────────────────────────────
#  LOAD IMAGES
# ─────────────────────────────────────────
IMG_PATH = os.path.join(os.path.dirname(__file__), "imgs")

BIRD_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join(IMG_PATH, "bird.png")))

PIPE_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join(IMG_PATH, "pipe.png")))

BG_IMG = pygame.transform.scale(
    pygame.image.load(os.path.join(IMG_PATH, "bg.png")),
    (WIN_W, WIN_H))

# ─────────────────────────────────────────
#  PYGAME INIT
# ─────────────────────────────────────────
pygame.init()
WIN   = pygame.display.set_mode((WIN_W, WIN_H))
CLOCK = pygame.time.Clock()
FONT  = pygame.font.SysFont("comicsans", 30)
pygame.display.set_caption("Flappy Bird NEAT")


# ═════════════════════════════════════════
#  BIRD CLASS
# ═════════════════════════════════════════
class Bird:
    """
    The bird the AI controls.
    - jump()  → snap velocity upward
    - move()  → apply gravity every frame
    - draw()  → render sprite, tilted by velocity
    """
    MAX_VEL = 10     # max downward speed
    GRAVITY  = 1.5   # speed added per frame

    def __init__(self, x, y):
        self.x   = x
        self.y   = float(y)
        self.vel = 0

    def jump(self):
        self.vel = -10.5   # negative = upward in pygame coords

    def move(self):
        self.vel  = min(self.vel + self.GRAVITY, self.MAX_VEL)
        self.y   += self.vel

    def get_mask(self):
        return pygame.mask.from_surface(BIRD_IMG)

    def draw(self, win):
        # Tilt up when rising, tilt down when falling
        tilt = max(-90, min(45, -self.vel * 3))
        rotated = pygame.transform.rotate(BIRD_IMG, tilt)
        # Keep the rotated image centered on the bird
        rect = rotated.get_rect(
            center=BIRD_IMG.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated, rect.topleft)


# ═════════════════════════════════════════
#  PIPE CLASS
# ═════════════════════════════════════════
class Pipe:
    """
    A pair of pipes (top + bottom) with a random gap.
    - move()     → scroll left every frame
    - draw()     → draw both pipes
    - collide()  → pixel-perfect collision with a bird
    """
    GAP = 200   # vertical space between the two pipes
    VEL =   5   # scroll speed in pixels per frame

    def __init__(self, x):
        self.x      = x
        self.height = random.randrange(80, 450)
        self.top    = self.height - PIPE_IMG.get_height()
        self.bottom = self.height + self.GAP
        self.passed = False

        # Flip the pipe image upside-down for the top pipe
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(PIPE_IMG,      (self.x, self.bottom))

    def collide(self, bird):
        """
        Uses pygame masks for pixel-perfect collision.
        Returns True if the bird overlaps with either pipe.
        """
        bird_mask = bird.get_mask()
        top_mask  = pygame.mask.from_surface(self.PIPE_TOP)
        bot_mask  = pygame.mask.from_surface(PIPE_IMG)

        top_offset = (self.x - int(bird.x), self.top    - round(bird.y))
        bot_offset = (self.x - int(bird.x), self.bottom - round(bird.y))

        return (bird_mask.overlap(top_mask, top_offset) or
                bird_mask.overlap(bot_mask, bot_offset))


# ═════════════════════════════════════════
#  DRAW WINDOW
# ═════════════════════════════════════════
def draw_window(win, birds, pipes, score, generation):
    """
    Redraws every element each frame.
    Order: background → pipes → birds → UI text
    """
    # 1. Background (bottom layer)
    win.blit(BG_IMG, (0, 0))

    # 2. Pipes
    for pipe in pipes:
        pipe.draw(win)

    # 3. Birds (drawn on top of pipes)
    for bird in birds:
        bird.draw(win)

    # 4. Score — top left
    score_label = FONT.render(f"Score: {score}", 1, (255, 255, 255))
    win.blit(score_label, (10, 10))

    # 5. Bird count — top right
    alive_label = FONT.render(f"Birds: {len(birds)}", 1, (255, 255, 255))
    win.blit(alive_label, (WIN_W - 10 - alive_label.get_width(), 10))

    # 6. Generation — below bird count
    gen_label = FONT.render(f"Gen: {generation}", 1, (255, 255, 255))
    win.blit(gen_label, (WIN_W - 10 - gen_label.get_width(), 45))

    # 7. Push the finished frame to the screen
    pygame.display.update()


# ═════════════════════════════════════════
#  EVAL GENOMES  (called by NEAT each gen)
# ═════════════════════════════════════════
GEN = 0   # generation counter (used for display only)

def eval_genomes(genomes, config):
    """
    Runs the game once with all birds in this generation
    playing simultaneously.  NEAT calls this function
    automatically each generation.
    """
    global GEN
    GEN += 1

    # ── Build parallel lists: one entry per bird ──────────
    nets   = []   # neural network for each bird
    birds  = []   # Bird object for each genome
    ge     = []   # genome for each bird (to set fitness)

    for _, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        ge.append(genome)

    pipes = [Pipe(700)]
    score = 0

    # ── Main loop ─────────────────────────────────────────
    while len(birds) > 0:
        CLOCK.tick(FPS)

        # Quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # ── Which pipe are the birds heading toward? ──────
        # If the leading edge of the first pipe is behind the
        # bird AND there is a second pipe, target that one.
        pipe_idx = 0
        if len(pipes) > 1 and birds[0].x > pipes[0].x + 60:
            pipe_idx = 1

        # ── Move each bird & ask its brain to jump or not ─
        for i, bird in enumerate(birds):
            bird.move()
            ge[i].fitness += 0.1   # reward just for staying alive

            # Three inputs fed to the neural network:
            #   1. Bird's Y position
            #   2. Distance to the top pipe opening
            #   3. Distance to the bottom pipe opening
            output = nets[i].activate((
                bird.y,
                abs(bird.y - pipes[pipe_idx].height),
                abs(bird.y - pipes[pipe_idx].bottom),
            ))

            # Output > 0.5 means the network says "jump"
            if output[0] > 0.5:
                bird.jump()

        # ── Move pipes, check pass & collision ────────────
        pipes_to_remove = []

        for pipe in pipes:
            pipe.move()

            # Check collision for every remaining bird
            for i in reversed(range(len(birds))):
                if pipe.collide(birds[i]):
                    ge[i].fitness -= 1     # punish crashing
                    birds.pop(i)
                    nets.pop(i)
                    ge.pop(i)

            # Did the birds just pass this pipe?
            if (not pipe.passed
                    and birds
                    and birds[0].x > pipe.x + 60):
                pipe.passed  = True
                score       += 1
                for g in ge:
                    g.fitness += 5         # big reward for clearing a pipe
                pipes.append(Pipe(700))    # spawn the next pipe

            # Remove pipes that have scrolled off the left edge
            if pipe.x + PIPE_IMG.get_width() < 0:
                pipes_to_remove.append(pipe)

        for pipe in pipes_to_remove:
            pipes.remove(pipe)

        # ── Remove birds that hit the floor or ceiling ────
        for i in reversed(range(len(birds))):
            if (birds[i].y + BIRD_IMG.get_height() >= WIN_H
                    or birds[i].y < 0):
                birds.pop(i)
                nets.pop(i)
                ge.pop(i)

        # ── Draw everything ───────────────────────────────
        draw_window(WIN, birds, pipes, score, GEN)


# ═════════════════════════════════════════
#  RUN NEAT
# ═════════════════════════════════════════
def run_neat(config_path):
    """
    Loads the NEAT config, creates a population,
    and starts the evolutionary loop.
    """
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)

    # Print progress to the terminal every generation
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    # Run for up to 50 generations; stop early if a bird
    # reaches the fitness_threshold set in the config file.
    winner = population.run(eval_genomes, 50)

    print(f"\nBest genome fitness: {winner.fitness:.2f}")


# ═════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════
if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(__file__),
        "config-feedforward.txt"
    )
    run_neat(config_path)