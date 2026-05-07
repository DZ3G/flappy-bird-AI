import pygame, neat, os, random

# Window settings
WIN_W = 500
WIN_H = 800
FPS   = 30

pygame.init()
WIN = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("Flappy Bird NEAT")
CLOCK = pygame.time.Clock()

# Load your PNG images from the imgs/ folder
PATH = os.path.join(os.path.dirname(__file__), "imgs")

BIRD_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join(PATH, "bird.png")))

PIPE_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join(PATH, "pipe.png")))

BG_IMG = pygame.transform.scale(
    pygame.image.load(os.path.join(PATH, "bg.png")),
    (WIN_W, WIN_H))  # stretch bg to fill window

FONT = pygame.font.SysFont("comicsans", 30)

# 1. Load NEAT config from your file
def run_neat(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # 2. Create a population of 50 birds
    pop = neat.Population(config)

    # 3. Add a reporter so you can see progress
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # 4. Run the game — eval_genomes is called each gen
    winner = pop.run(eval_genomes, 50)
    print(f"Best bird score: {winner.fitness}")


# 5. This runs once per generation with all 50 birds
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0   # start score at 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # → run game, increase genome.fitness as bird survives


# Entry point
if __name__ == "__main__":
    cfg = os.path.join(os.path.dirname(__file__),
                       "config-feedforward.txt")
    run_neat(cfg)
