# 🐦 Flappy Bird AI — Tutoriel Complet en Français

> **Pour qui ?** Ce tutoriel est écrit pour quelqu'un qui commence en programmation.
> Chaque étape est expliquée simplement, sans rien sauter.

---

## 📋 Table des matières

1. [C'est quoi ce projet ?](#1-cest-quoi-ce-projet-)
2. [Comment ça marche — l'idée générale](#2-comment-ça-marche--lidée-générale)
3. [Structure des fichiers](#3-structure-des-fichiers)
4. [Installation — préparer ton ordinateur](#4-installation--préparer-ton-ordinateur)
5. [Le fichier Main.py — explication ligne par ligne](#5-le-fichier-mainpy--explication-ligne-par-ligne)
6. [Le fichier config-feedforward.txt — régler l'IA](#6-le-fichier-config-feedforwardtxt--régler-lia)
7. [Lancer le projet](#7-lancer-le-projet)
8. [Ce que tu vas voir à l'écran](#8-ce-que-tu-vas-voir-à-lécran)
9. [Comprendre NEAT en 2 minutes](#9-comprendre-neat-en-2-minutes)
10. [Problèmes fréquents et solutions](#10-problèmes-fréquents-et-solutions)

---

## 1. C'est quoi ce projet ?

Ce projet fait jouer une **intelligence artificielle** au jeu Flappy Bird — toute seule, sans que tu lui expliques quoi faire.

Le principe :
- 50 oiseaux jouent en même temps
- Chaque oiseau a un petit **cerveau artificiel** (réseau neuronal)
- Les oiseaux qui survivent le plus longtemps ont les meilleurs cerveaux
- Ces cerveaux font des "bébés" légèrement différents
- Après environ 10 générations, un oiseau joue parfaitement

La technique utilisée s'appelle **NEAT** *(NeuroEvolution of Augmenting Topologies)* — c'est un algorithme inventé par des chercheurs de l'Université du Texas en 2002.

---

## 2. Comment ça marche — l'idée générale

```
Génération 1 :  50 oiseaux volent → tous meurent vite (cerveaux au hasard)
                         ↓
             Les meilleurs cerveaux sont gardés
                         ↓
Génération 2 :  50 nouveaux oiseaux (cerveaux légèrement améliorés)
                         ↓
                  ... ça recommence ...
                         ↓
Génération ~10 : Un oiseau passe tous les tuyaux sans jamais mourir 🏆
```

### Ce que le cerveau de chaque oiseau "voit" (les 3 entrées)

```
┌─────────────────────────────────────────────┐
│                                             │
│   1. Hauteur de l'oiseau (bird.y)           │
│                                             │
│   2. Distance entre l'oiseau               │
│      et le HAUT du tuyau                   │
│                                             │
│   3. Distance entre l'oiseau               │
│      et le BAS du tuyau                    │
│                                             │
└─────────────────────────────────────────────┘
              ↓
       Cerveau (NEAT)
              ↓
   Sortie > 0.5 → SAUTER
   Sortie < 0.5 → ne rien faire
```

### Comment le score (fitness) est calculé

| Événement | Points |
|-----------|--------|
| Survivre 1 frame (1/30ème de seconde) | +0.1 |
| Passer un tuyau | +5.0 |
| Toucher un tuyau ou le sol | -1.0 |

---

## 3. Structure des fichiers

```
flappy-bird-AI/
│
├── Main.py                  ← le programme principal (jeu + IA)
├── config-feedforward.txt   ← les réglages de l'IA (NEAT)
├── README.md                ← ce fichier
│
└── IMGs/                    ← les images du jeu
    ├── Bird.png             ← l'oiseau
    ├── pipe.png             ← les tuyaux
    ├── bg.png               ← le fond d'écran
    └── base.png             ← le sol
```

> ⚠️ **Important :** Le dossier `IMGs` doit être dans le même dossier que `Main.py`.
> Si les images ne sont pas au bon endroit, le programme plante au démarrage.

---

## 4. Installation — préparer ton ordinateur

### Étape 1 — Vérifier que Python est installé

Ouvre un terminal (sur Windows : touche Windows → tape "cmd" → Entrée) et tape :

```bash
python --version
```

Tu dois voir quelque chose comme `Python 3.10.x`. Si tu as une erreur, télécharge Python sur [python.org](https://python.org).

### Étape 2 — Installer les bibliothèques nécessaires

Dans le terminal, tape cette commande :

```bash
pip install pygame neat-python
```

Voilà ce que fait chaque bibliothèque :

| Bibliothèque | Rôle |
|--------------|------|
| `pygame` | Affiche la fenêtre du jeu, gère les images et les collisions |
| `neat-python` | Fait évoluer les cerveaux des oiseaux génération après génération |

### Étape 3 — Télécharger le projet

Si tu utilises GitHub Desktop, clique sur **Clone repository** et colle l'adresse du repo.

Ou avec le terminal :

```bash
git clone https://github.com/TON_NOM/flappy-bird-AI
cd flappy-bird-AI
```

---

## 5. Le fichier Main.py — explication ligne par ligne

### 5.1 Les imports et constantes

```python
import pygame      # pour afficher le jeu
import neat        # pour l'intelligence artificielle
import os          # pour trouver les fichiers sur le disque
import random      # pour la hauteur aléatoire des tuyaux

WIN_W = 500        # largeur de la fenêtre en pixels
WIN_H = 800        # hauteur de la fenêtre en pixels
FPS   = 30         # images par seconde (vitesse du jeu)
```

### 5.2 Le chargement des images

```python
IMG_PATH = os.path.join(os.path.dirname(__file__), "imgs")
# → trouve le dossier "imgs" là où est le fichier Python

BIRD_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join(IMG_PATH, "bird.png")))
# → charge bird.png et le double de taille (scale2x)
```

> 💡 `scale2x` double la taille de l'image. Sans ça, les sprites seraient trop petits.

### 5.3 La classe Bird (l'oiseau)

La classe `Bird` gère toute la physique de l'oiseau.

```python
class Bird:
    MAX_VEL = 10      # vitesse maximale vers le bas
    GRAVITY  = 1.5    # gravité ajoutée chaque frame

    def __init__(self, x, y):
        self.x   = x        # position horizontale (fixe)
        self.y   = float(y) # position verticale (change à chaque frame)
        self.vel = 0         # vitesse verticale (0 = immobile)
```

**La méthode `jump()`** — appelée quand l'IA décide de sauter :

```python
def jump(self):
    self.vel = -10.5
    # Négatif car en pygame, y=0 est en HAUT de l'écran
    # Une vitesse négative fait monter l'oiseau
```

**La méthode `move()`** — appelée chaque frame pour simuler la gravité :

```python
def move(self):
    self.vel  = min(self.vel + self.GRAVITY, self.MAX_VEL)
    # → ajoute la gravité à la vitesse, mais jamais plus que MAX_VEL
    self.y   += self.vel
    # → déplace l'oiseau selon sa vitesse
```

**La méthode `get_mask()`** — pour les collisions précises :

```python
def get_mask(self):
    return pygame.mask.from_surface(BIRD_IMG)
    # → crée un masque de pixels de la forme exacte de l'oiseau
    # → bien mieux qu'un rectangle pour détecter les collisions
```

### 5.4 La classe Pipe (les tuyaux)

```python
class Pipe:
    GAP = 200   # espace entre le tuyau haut et le tuyau bas
    VEL =   5   # vitesse de déplacement vers la gauche

    def __init__(self, x):
        self.height = random.randrange(80, 450)
        # → hauteur aléatoire du gap entre 80px et 450px depuis le haut
        self.top    = self.height - PIPE_IMG.get_height()
        self.bottom = self.height + self.GAP
        self.passed = False   # l'oiseau a-t-il déjà passé ce tuyau ?

        # Le tuyau du haut est l'image retournée à l'envers
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
```

**La méthode `collide()`** — collision pixel par pixel :

```python
def collide(self, bird):
    bird_mask = bird.get_mask()
    top_mask  = pygame.mask.from_surface(self.PIPE_TOP)
    bot_mask  = pygame.mask.from_surface(PIPE_IMG)

    # offset = différence de position entre l'oiseau et le tuyau
    top_offset = (self.x - int(bird.x), self.top    - round(bird.y))
    bot_offset = (self.x - int(bird.x), self.bottom - round(bird.y))

    # overlap() renvoie None si pas de collision
    return (bird_mask.overlap(top_mask, top_offset) or
            bird_mask.overlap(bot_mask, bot_offset))
```

> 💡 La collision par masque est plus précise qu'un rectangle. Elle regarde pixel par pixel si les deux images se touchent vraiment.

### 5.5 La fonction draw_window()

Redessine tout l'écran à chaque frame, dans le bon ordre :

```python
def draw_window(win, birds, pipes, score, generation):
    win.blit(BG_IMG, (0, 0))      # 1. fond (couche la plus basse)
    for pipe in pipes:
        pipe.draw(win)             # 2. tuyaux
    for bird in birds:
        bird.draw(win)             # 3. oiseaux (par-dessus les tuyaux)
    # 4. texte (Score, Birds, Gen) par-dessus tout
    pygame.display.update()        # 5. envoyer l'image à l'écran
```

> ⚠️ L'ordre compte ! Si tu dessines les oiseaux avant le fond, tu ne les vois pas.

### 5.6 La fonction eval_genomes() — le cœur de l'IA

C'est la fonction la plus importante. NEAT l'appelle automatiquement à chaque génération.

```python
def eval_genomes(genomes, config):
    global GEN
    GEN += 1

    # 3 listes toujours synchronisées — même index = même oiseau
    nets   = []   # cerveau (réseau neuronal) de chaque oiseau
    birds  = []   # objet Bird de chaque oiseau
    ge     = []   # génome NEAT de chaque oiseau (pour le score)

    for _, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))   # tous au même point de départ
        ge.append(genome)
```

**La décision de sauter — chaque frame, pour chaque oiseau :**

```python
output = nets[i].activate((
    bird.y,                              # où est l'oiseau ?
    abs(bird.y - pipes[pipe_idx].height),  # distance au tuyau haut
    abs(bird.y - pipes[pipe_idx].bottom),  # distance au tuyau bas
))

if output[0] > 0.5:
    bird.jump()   # le cerveau dit "saute !"
```

**Quand un oiseau touche un tuyau :**

```python
if pipe.collide(birds[i]):
    ge[i].fitness -= 1      # -1 point (pénalité)
    birds.pop(i)            # retirer l'oiseau
    nets.pop(i)             # retirer son cerveau
    ge.pop(i)               # retirer son génome
    # ⚠️ Les 3 listes doivent être retirées ensemble !
```

**Quand un oiseau passe un tuyau :**

```python
if not pipe.passed and birds and birds[0].x > pipe.x + 60:
    pipe.passed = True
    score += 1
    for g in ge:
        g.fitness += 5    # +5 pour TOUS les oiseaux encore vivants
    pipes.append(Pipe(700))  # créer un nouveau tuyau
```

### 5.7 La fonction run_neat() — démarrer l'entraînement

```python
def run_neat(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path          # charge config-feedforward.txt
    )

    population = neat.Population(config)   # crée 50 oiseaux
    population.add_reporter(neat.StdOutReporter(True))  # affiche les stats dans le terminal

    winner = population.run(eval_genomes, 50)
    # → lance 50 générations maximum
    # → NEAT appelle eval_genomes() automatiquement à chaque génération
    # → s'arrête si un oiseau atteint fitness_threshold = 100
```

---

## 6. Le fichier config-feedforward.txt — régler l'IA

Ce fichier contrôle comment NEAT évolue. Tu n'as pas besoin de tout comprendre, mais voici les lignes les plus importantes :

```ini
[NEAT]
fitness_criterion       = max       # garder le meilleur score
fitness_threshold       = 100       # s'arrêter quand un oiseau atteint 100 pts
pop_size                = 50        # 50 oiseaux par génération
no_fitness_termination  = False     # s'arrêter quand fitness_threshold est atteint

[DefaultGenome]
num_inputs              = 3         # 3 entrées : hauteur, dist haut, dist bas
num_outputs             = 1         # 1 sortie : sauter ou non
num_hidden              = 0         # commence sans couche cachée (NEAT en ajoute si besoin)

[DefaultReproduction]
elitism                 = 2         # les 2 meilleurs survivent sans mutation
survival_threshold      = 0.2       # seul le top 20% peut se reproduire

[DefaultStagnation]
max_stagnation          = 20        # supprimer une espèce si pas de progrès en 20 gens
```

> 💡 **Pour aller plus vite :** augmente `pop_size` à 100. Tu auras plus d'oiseaux donc l'IA apprend plus vite, mais chaque génération sera un peu plus lente à afficher.

---

## 7. Lancer le projet

### Dans VS Code

1. Ouvre le dossier `flappy-bird-AI` dans VS Code
2. Ouvre `Main.py`
3. Appuie sur **F5** ou fais clic droit → **Run Python File**

### Dans le terminal

```bash
cd chemin/vers/flappy-bird-AI
python Main.py
```

---

## 8. Ce que tu vas voir à l'écran

```
┌──────────────────────────┐
│ Score: 0    Birds: 50    │
│             Gen: 1       │
│                          │
│  🐦🐦🐦🐦🐦               │
│  🐦🐦🐦🐦🐦    ║   ║      │
│  🐦🐦🐦🐦🐦    ║   ║      │
│             ║   ║      │
│  🐦🐦🐦🐦🐦               │
│  🐦🐦🐦🐦🐦    ║   ║      │
└──────────────────────────┘
```

| Affichage | Signification |
|-----------|---------------|
| `Score` | Nombre de tuyaux passés par le meilleur oiseau |
| `Birds` | Nombre d'oiseaux encore en vie cette génération |
| `Gen` | Numéro de la génération en cours |

### Progression typique

| Génération | Ce qui se passe |
|------------|-----------------|
| 1–2 | Tous les oiseaux meurent en quelques secondes |
| 3–5 | Certains oiseaux commencent à sauter au bon moment |
| 6–10 | Un oiseau passe 1 ou 2 tuyaux |
| 10–15 | Un oiseau passe tous les tuyaux sans jamais mourir 🏆 |

### Dans le terminal

À chaque génération, NEAT affiche des stats :

```
****** Running generation 0 ******
Population's average fitness: 2.34
Best fitness: 12.50 - size (3, 3) - species 1

****** Running generation 1 ******
Population's average fitness: 8.71
Best fitness: 31.20 - size (3, 4) - species 1
```

---

## 9. Comprendre NEAT en 2 minutes

NEAT = **NeuroEvolution of Augmenting Topologies**

C'est un algorithme qui fait évoluer des réseaux de neurones comme si c'était de la vraie évolution biologique.

### Les 3 idées clés de NEAT

**1. Partir de zéro**
Le cerveau de chaque oiseau commence minimaliste : juste 3 entrées connectées à 1 sortie. NEAT ajoute des neurones seulement quand c'est utile. C'est 25× plus rapide que de partir d'un gros cerveau au hasard.

**2. La spéciation (les espèces)**
Les oiseaux avec des cerveaux similaires sont regroupés en "espèces". Chaque espèce concourt séparément. Ça protège les nouvelles idées bizarres qui ont besoin de temps pour s'améliorer avant d'être jugées.

**3. Les innovation numbers**
Quand deux cerveaux se croisent pour faire un "bébé", NEAT sait exactement quels neurones correspondent à quels autres grâce à des numéros uniques. Ça évite le problème du "quelle pièce va avec quelle pièce".

### La formule du fitness dans ce projet

```
fitness = (frames_survécues × 0.1)
        + (tuyaux_passés   × 5.0)
        - (collisions      × 1.0)

Exemple : 200 frames survécues + 3 tuyaux passés - 1 collision
        = (200 × 0.1) + (3 × 5) - (1 × 1)
        = 20 + 15 - 1
        = 34.0 pts
```

---

## 10. Problèmes fréquents et solutions

### ❌ `ModuleNotFoundError: No module named 'pygame'`

```bash
pip install pygame
```

### ❌ `ModuleNotFoundError: No module named 'neat'`

```bash
pip install neat-python
```

> ⚠️ Le paquet s'appelle `neat-python` à l'installation, mais `import neat` dans le code.

### ❌ `Missing required configuration item: 'no_fitness_termination'`

Ouvre `config-feedforward.txt` et ajoute cette ligne dans `[NEAT]` :

```ini
no_fitness_termination = False
```

### ❌ Écran noir au démarrage

Vérifie que le dossier `IMGs` contient bien `Bird.png`, `pipe.png`, `bg.png`, `base.png`.
Les noms sont sensibles à la casse — `Bird.png` ≠ `bird.png` sur certains systèmes.

### ❌ `python` non reconnu dans le terminal

Essaie `python3` à la place :

```bash
python3 Main.py
```

### ❌ VS Code utilise le mauvais Python

1. `Ctrl+Shift+P`
2. Tape **Python: Select Interpreter**
3. Choisis le Python qui a tes packages installés

---

## 📚 Sources et ressources

- **Papier original de NEAT** — Stanley & Miikkulainen, 2002 : [nn.cs.utexas.edu](https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf)
- **Documentation neat-python** : [neat-python.readthedocs.io](https://neat-python.readthedocs.io)
- **Tutoriel vidéo de référence** : [Tech With Tim — YouTube](https://www.youtube.com/watch?v=OGHA-elMrxI)
- **pygame** : [pygame.org/docs](https://www.pygame.org/docs/)

---

*README rédigé dans le cadre d'un projet scolaire — Flappy Bird AI avec NEAT*