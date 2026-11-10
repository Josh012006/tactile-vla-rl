# Environnement Isaac Lab : FORGE PegInsert avec Robotiq 2F-85 et taxels

Ce document décrit l'environnement de simulation de `tactile-vla-rl` : ce qu'il contient, comment le lancer et le piloter, ce qu'il coûte en temps et en mémoire GPU quand on augmente le nombre d'environnements, et surtout **ce qui est garanti et ce qui ne l'est pas** en matière de reproductibilité.

Tout ce qui est chiffré ici vient de mesures faites avec les tests du dépôt (`tests/test_isaac_*.py`) sur un **NVIDIA L40S (46 GB), Isaac Sim 5.1, Isaac Lab 2.3.0, torch 2.7.0**. Rien n'est affirmé au-delà de ce qui a été mesuré ; ce qui ne l'a pas été est listé dans la section 9.

---

## 1. En bref

| | |
|---|---|
| Tâche | `Isaac-Forge-PegInsert-Direct-v0` (FORGE), consigne : « Insert the peg into the socket. » |
| Robot par défaut | Franka + Robotiq 2F-85 + **56 taxels** (2 doigts × 7 × 4), profil `robotiq_2f85_tsf` |
| Actions | 8 valeurs : 7 cibles articulaires **absolues** (rad) + commande de pince (0 ouvert, 1 fermé) |
| Observations | 2 images 224×224, joints, pince, force/couple, offset cheville–socle, effort de pince, **tactile (2, 7, 4) N** |
| Cadence | 15 Hz de contrôle (décimation 8 à 120 Hz de physique), épisodes de 150 pas au plus |
| Parallélisme | N environnements indépendants dans **une** scène et **un** processus, pilotés par « rounds » |
| Interface | serveur WebSocket + msgpack, un aller-retour par pas, client `VectorEnvClient` (numpy seulement) |

**Reproductibilité en une table** (détails section 7) :

| Garanti (mesuré) | Non garanti (mesuré) |
|---|---|
| La **même séquence** de `reset` / `step`, rejouée depuis un serveur neuf, donne des observations **identiques bit à bit** (tactile compris) | Le départ d'un `(seed, round)` **après un autre épisode** (Robotiq uniquement) : environ 1 environnement sur 3 démarre un peu ailleurs, médiane 0,01 à 0,06 mm sur la cheville, rarement 1 à 11 mm |
| Les environnements d'un round sont **exactement indépendants** les uns des autres | Le départ dépend aussi de `num_envs` et de l'indice de l'environnement |
| Le reset est identique bit à bit **jusqu'à la fermeture de la pince** sur la cheville | Les images (rendu) n'ont pas été comparées bit à bit |
| Avec le Franka, le départ ne dépend **pas** de l'épisode précédent | Un autre GPU, pilote ou version d'Isaac n'a pas été testé |

---

## 2. Ce que l'environnement intègre

**Tâche.** L'insertion de cheville de FORGE (Isaac Lab), avec la vraie logique de reset de FORGE : elle agit sur **tous** les environnements et fait avancer la physique, ce qui impose le fonctionnement par rounds (section 5).

**Robots** (champ `robot` du YAML) :

| Profil | Contenu |
|---|---|
| `franka` | Franka + Franka Hand de FORGE, sans modification |
| `robotiq_2f85` | Bras Franka + Robotiq 2F-85 (USD NVIDIA RoboLab), sans taxel |
| `robotiq_2f85_tsf` | Idem + 7×4 zones tactiles par doigt (56 taxels), envoie `extra/tactile` en newtons |

L'USD du Robotiq est construit depuis `third_party/robotiq/` au **premier démarrage** du serveur (`src/envs/isaaclab/robot_assets.py`, résultat dans `third_party/robotiq/_build/`). Le profil Robotiq porte `finger_effort = 6.0` et `fingerpad_length = 0.0056`.

**Capteurs envoyés en permanence.** Le serveur envoie tout ce que le robot peut mesurer (tactile, force, effort de pince) ; c'est le consommateur qui choisit quoi utiliser. Un seul serveur sert donc les runs avec et sans toucher.

**Dynamique.** Gains PD du bras 400 / 60 (FORGE les met à zéro), timeout d'Isaac Lab désactivé (le suivi d'épisode est fait côté serveur, par environnement), solveur PhysX GPU déterministe (`enhanced_determinism: true`).

**Épisodes.** Récompense creuse 1.0 là où FORGE dit « inserted » (xy < 2,5 mm et z sous son seuil), `terminated` = succès (maintenu `success_hold_steps` pas), `truncated` = timeout à `max_steps_per_episode`. Par défaut la pince ne s'ouvre pas (`gripper_can_open: false` : `action[:, 7]` est ignoré, la pince reste à la largeur de saisie du reset), car le pi05_droid_jointpos zero-shot a tendance à ouvrir et à lâcher la cheville.

### Où est le code

```
src/envs/                     logique commune à tous les environnements (aucun import de simulateur ni de torch)
  protocol.py                 contrat serveur/client : rounds, clés d'observation, EnvSpec, StepResult
  wire.py  client.py          msgpack + VectorEnvClient (un ou plusieurs serveurs vus comme un seul vecteur)
  collect.py                  Collector (modes steps / episodes) et run_fixed_rounds (évaluation sur un ensemble fixe)
  episode_manager.py  episode_tracker.py  backend.py  ws_server.py     rounds, suivi d'épisode, serveur WebSocket
src/envs/isaaclab/            tout ce qui est propre à Isaac Lab (importé seulement côté simulateur)
  server.py                   point d'entrée : python -m src.envs.isaaclab.server --config ...
  forge_backend.py            adaptateur simulateur : reset, step, observe, outils de debug, surcharges PhysX
  forge_env.py                ForgeEnvJointPosPi05 : espace d'action, adaptateur Robotiq, taxels, reset_lockstep
  robots.py  robot_assets.py  profils de robot, construction de l'USD
configs/task/peg_insert_forge.yaml    la configuration de la tâche (seul fichier lu par le serveur)
jobs/env_server.sh  jobs/job_isaac_tests.sh  scripts/setup_isaac_venv.sh
tests/                        tests CPU (fakes) et tests Isaac (--isaac, GPU)
```

Les fichiers de `src/envs/` n'importent jamais `src/envs/isaaclab/` : c'est vérifié par `tests/test_static.py`. Le côté apprentissage (JAX) n'a donc besoin ni d'Isaac Sim ni de torch.

---

## 3. Installation

Deux environnements virtuels séparés :

| Venv | Rôle | Création |
|---|---|---|
| `.venv` | côté apprentissage + client (numpy, msgpack, websockets, pyyaml) | `uv sync` |
| `.venv-isaaclab` | simulateur : Isaac Sim 5.1, torch 2.7.0 (cu128), Isaac Lab (submodule `third_party/IsaacLab`) | `bash scripts/setup_isaac_venv.sh` (une fois, 10 à 20 min, ~10 GB, pas de GPU nécessaire) |

Le script est idempotent ; il initialise le submodule `third_party/IsaacLab` si besoin. `third_party/robotiq/` n'est **pas** un submodule : ce sont les fichiers originaux de NVIDIA, copiés à la main et listés dans `.gitignore` (jamais commités) : `robotiq_usd/franka_robotiq_2f_85_flattened.usd`, `robotiq_usd/TSF_85.usd` et `robotiq-tsf85-contact-twin-main/`. Sans eux, le serveur s'arrête avec un `FileNotFoundError` qui les liste.

Tests CPU, sans simulateur :

```bash
uv run pytest tests -q          # les tests Isaac sont ignorés (skipped) sans --isaac
```

---

## 4. Démarrer un serveur

Sur un nœud GPU (L40S) :

```bash
jobs/env_server.sh                                          # configs/task/peg_insert_forge.yaml, 8 environnements, port 8102
jobs/env_server.sh --set num_envs=16 --set server_port=8103  # surcharge d'un champ du YAML
ENV_CONFIG=autre.yaml jobs/env_server.sh                     # autre fichier de configuration
```

Le script active `.venv-isaaclab`, exporte `OMNI_KIT_ACCEPT_EULA=YES` et `CUBLAS_WORKSPACE_CONFIG=:4096:8` (nécessaire à la reproductibilité, voir section 7), puis lance `python -m src.envs.isaaclab.server`. Le serveur affiche `[env_server] ready: N environments, port ...` quand il accepte des connexions. Démarrage mesuré : 25 à 55 s (N = 1 à 64), plus la construction de l'USD au tout premier démarrage.

- **Plusieurs serveurs** (un par GPU) : démarrer le premier seul et attendre `ready` : le premier démarrage écrit l'USD dans `third_party/robotiq/_build/` et deux premiers démarrages simultanés l'écriraient en même temps.
- Arrêt propre sur SIGINT / SIGTERM.
- Le client n'attend un serveur qui démarre que lors de la première connexion.

### Champs du YAML (`configs/task/peg_insert_forge.yaml`)

| Champ | Défaut | Effet |
|---|---|---|
| `num_envs` | 8 | environnements simulés par ce serveur |
| `clone_in_fabric` | false | **doit rester false** : sinon caméras et capteurs de contact ne voient qu'un environnement |
| `robot` | `robotiq_2f85_tsf` | `franka` / `robotiq_2f85` / `robotiq_2f85_tsf` |
| `env_seed`, `auto_reset` | 0, true | séquence de rounds par défaut ; prochain round lancé automatiquement |
| `camera_size` | 224 | côté des images (carré) ; l'entrée du VLA est 224×224 |
| `max_steps_per_episode`, `control_hz` | 150, 15 | timeout par environnement ; `control_hz` est informatif (réel : décimation 8 à 120 Hz) |
| `success_hold_steps` | 1 | pas consécutifs « inserted » pour déclarer le succès (≥ 1) |
| `gripper_can_open`, `gripper_obs` | false, `normalized` | pince figée ou non ; `gripper_position` = `1 − largeur/0,04` (0 ouvert, 1 fermé, ~0,9 en tenant la cheville), ou `closed` / `raw` |
| `arm_stiffness`, `arm_damping` | 400, 60 | gains PD du bras |
| `disable_sleep` | false | (n'a pas d'effet mesuré sur la reproductibilité) |
| `enhanced_determinism` | true | solveur PhysX GPU déterministe ; **identique pour toutes les données d'une expérience** |
| `reset_passes` | 2 | nombre de resets enchaînés à chaque `reset` (chacun coûte un vrai reset) |
| `finger_stiffness`, `finger_damping`, `finger_effort`, `fingerpad_length`, `robot_usd`, `tactile_exclude_zone_joints`, `reset_grip_open_width` | (commentés) | réglages Robotiq optionnels |
| `physx_<champ>` (via `--set`) | | n'importe quel champ de `PhysxCfg` d'Isaac Lab, converti au type du champ |
| `server_host`, `server_port` | `0.0.0.0`, 8102 | |

Les champs de dynamique et de reproductibilité demandent un redémarrage du serveur.

---

## 5. Utiliser l'environnement

### Protocole en rounds

Le reset de FORGE agit sur tous les environnements et fait avancer la physique : les environnements d'un serveur **commencent donc leurs épisodes ensemble** (un *round*, avec le vrai reset de FORGE). Chacun déroule ensuite son propre épisode (succès ou timeout à son rythme). Un environnement dont l'épisode est fini est **maintenu en place** (`acting = False`) jusqu'à ce que les autres aient fini ; le round suivant démarre alors.

- `reset(seed, round)` : lance un round. `seed` fixe toute la séquence de rounds, `round` la position dans la séquence (graine de simulation `seed + round`). Sans argument : round suivant. Avec `seed` seul : round 0.
- `step(actions)` : avance **tous** les environnements ; actions `(num_envs, 8)` float32. Quand le dernier environnement en cours finit et que `auto_reset` est actif, **le round suivant démarre dans le même appel** : `obs` est la première observation du nouveau round, la vraie dernière observation de chaque épisode terminé à ce pas est dans `final_obs` (lignes `final_ids`). Rien n'est perdu ni périmé.
- `auto_reset=False` : quand tous ont fini, rien ne repart seul, `step` est sans effet (`acting` faux partout) jusqu'au `reset` du client. C'est le mode de l'évaluation sur un ensemble fixe.
- Un pas coûte **un seul** aller-retour réseau. Aucune reprise silencieuse : si un appel échoue, l'exception remonte (relancer depuis un checkpoint plutôt que de masquer un épisode perdu).

Champs d'un `StepResult` : `obs`, `reward`, `terminated` (succès), `truncated` (timeout, pas un état terminal pour le bootstrap), `success`, `t`, `acting`, `done`, `all_done`, `round_id`, `final_obs`, `final_ids`.

### Observations

| Clé | Forme (un environnement) | Type | Contenu |
|---|---|---|---|
| `observation/exterior_image_1_left` | (224, 224, 3) | uint8 | caméra extérieure |
| `observation/wrist_image_left` | (224, 224, 3) | uint8 | caméra poignet |
| `observation/joint_position` | (7,) | float32 | joints du bras (rad) |
| `observation/gripper_position` | (1,) | float32 | selon `gripper_obs` |
| `extra/ft_force` | (3,) | float32 | force du capteur F/T de FORGE |
| `extra/force_threshold` | () | float32 | seuil de force de FORGE |
| `extra/peg_socket_offset` | (3,) | float32 | offset cheville–socle |
| `extra/gripper_effort` | () | float32 | effort moteur du doigt (Robotiq) |
| `extra/tactile` | (2, 7, 4) | float32 | N ; `[0]` doigt gauche, `[1]` droit, ligne 0 = bout du doigt (profil taxels seulement) |

Les clés sont celles de la convention DROID / pi05 (image + état) et de l'ancien serveur (`extra/*`).

### Exemple

```python
from src.envs.client import VectorEnvClient
from src.envs.collect import Collector, run_fixed_rounds

env = VectorEnvClient([("localhost", 8102)])             # ou plusieurs serveurs : [("node1", 8102), ("node2", 8102)]
res = env.reset(seed=0)                                  # round 0
while True:
    res = env.step(policy(res.obs))                      # policy : obs (N, ...) -> actions (N, 8) float32
    # res.done : épisodes finis à ce pas (dernière obs dans res.final_obs) ; res.acting faux = environnement en attente

# Collecte
collector = Collector(env, policy, mode="steps")         # Rollout (T, N, ...) ; ou mode="episodes" -> liste de Trajectory
data = collector.collect(80)                             # 80 pas dans CHAQUE environnement
update(data); collector.version += 1                     # pendant la mise à jour, les environnements ne sont pas avancés

# Évaluation sur un ensemble fixe de départs (auto_reset=False, aucun épisode coupé ni répété)
results = run_fixed_rounds(env, policy, rounds=range(8), seed=0)
# un dict par (round, environnement) : round, env, success, terminated, truncated, length, episode_return
```

Ce que le `Collector` garantit : `next_obs` d'une transition qui termine un épisode est la **dernière** observation de cet épisode (jamais la première du round suivant) ; `terminated` et `truncated` sont séparés ; chaque transition porte la `version` de politique qui a choisi l'action et son `round` ; les lignes `acting == False` ne sont pas des transitions (le mode `steps` les garde pour que les tableaux restent rectangulaires, le mode `episodes` ne les stocke jamais) ; les épisodes en cours à la fin d'un appel continuent à l'appel suivant. `total_idle_steps` compte les pas passés à attendre : c'est le prix des rounds synchrones.

Attention en mode `episodes` : s'arrêter au milieu d'un round garde les épisodes qui ont fini en premier, donc surreprésente les courts (les succès). Demander un multiple de `num_envs`, ou utiliser `run_fixed_rounds` pour évaluer.

---

## 6. Coûts et gains avec plusieurs environnements

Mesures : `COST_NUM_ENVS=1,4,16,64 SLURM_SUBMIT_DIR=$PWD bash jobs/job_isaac_tests.sh tests/test_isaac_cost.py` (L40S, profil `robotiq_2f85_tsf`, 150 pas par épisode, rollout T = 80 pas). Les pas sont mesurés sur des épisodes de 30 pas puis projetés à la vraie longueur. Rapport JSON : `reports/env_cost_<horodatage>.json`.

### Coût d'un environnement de plus

| N | VRAM (MiB) | pas (ms) | dont physique | dont observation | reset (s) | pas d'env / s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8273 | 519,9 | 505,9 | 11,4 | 8,19 | 2 |
| 4 | 8114 | 545,2 | 532,1 | 9,8 | 8,44 | 7 |
| 16 | 8453 | 578,4 | 559,7 | 12,0 | 9,22 | 28 |
| 64 | 9807 | 674,8 | 624,3 | 18,6 | 9,89 | 95 |

Ajustements linéaires sur ces quatre points :

| Grandeur | Partie fixe | Par environnement | R² |
|---|---:|---:|---:|
| VRAM | 8103 MiB | 26,3 MiB | 0,98 |
| pas | 530,8 ms | 2,29 ms | 0,98 |
| dont physique | 519,6 ms | 1,69 ms | 0,94 |
| dont observation | 10,2 ms | 0,13 ms | 0,95 |
| reset | 8,41 s | 0,025 s | 0,87 |

Lecture : **le coût est surtout fixe**. À N = 1, la physique pèse 97 % du pas et ce coût ne dépend presque pas de N ; la VRAM est dominée par ~8,1 GiB fixes (la VRAM mesurée à N = 4 est un peu sous celle de N = 1 : c'est du bruit de mesure, pas une économie). Ajouter des environnements dans le même processus est donc presque gratuit, alors que lancer plusieurs petits serveurs paierait ~8 GiB et ~0,5 s par pas à chaque fois (conclusion tirée du modèle, pas mesurée en multi-serveur).

### Gain

Mise à jour toutes les T = 80 pas de chaque environnement (`update_per: step`) :

| N | VRAM (MiB) | mise à jour (s) | échantillons / mise à jour | échantillons / s | gain | efficacité | part du reset | gain par GiB en plus |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8273 | 45,9 | 80 | 1,74 | 1,00 | 100 % | 10 % | — |
| 4 | 8114 | 48,2 | 320 | 6,63 | 3,81 | 95 % | 9 % | — |
| 16 | 8453 | 51,4 | 1280 | 24,91 | 14,30 | 89 % | 10 % | 75,6 |
| 64 | 9807 | 65,1 | 5120 | 78,64 | 45,15 | 71 % | 8 % | 29,5 |

Une mise à jour par round de N épisodes complets de 150 pas (`update_per: episode`, cible 64 épisodes) :

| N | VRAM (MiB) | mise à jour (s) | épisodes / mise à jour | épisodes / h | temps pour 64 épisodes (s) | gain | efficacité | gain par GiB en plus |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8273 | 86,0 | 1 | 41,9 | 5505 | 1,00 | 100 % | — |
| 4 | 8114 | 90,8 | 4 | 158,6 | 1453 | 3,79 | 95 % | — |
| 16 | 8453 | 94,8 | 16 | 607,4 | 379 | 14,51 | 91 % | 76,8 |
| 64 | 9807 | 111,5 | 64 | 2066,4 | 111 | 49,37 | 77 % | 32,3 |

« Efficacité » = gain / N. À N = 64 : ×45 d'échantillons par seconde, ×49 d'épisodes par heure, pour +1,5 GiB de VRAM par rapport à N = 1. Les épisodes sont comptés à 150 pas : dans la réalité, un environnement qui réussit tôt attend les autres (pas inactifs), ce que ces tableaux ne modélisent pas ; `Collector.total_idle_steps` le mesure sur les vraies données.

### Limites de ces chiffres

- **Mesuré seulement à N = 1, 4, 16, 64.** Rien au-delà de 64 ; le gain par GiB baisse déjà (76 → 32), ne pas extrapoler à N grand sans mesurer (`COST_NUM_ENVS=...,128`).
- `max_envs_in_gpu_memory = 1268` est l'extrapolation de l'ajustement de VRAM avec une marge de 10 % : **ce n'est pas une mesure**, et le temps de pas augmenterait aussi.
- Une seule machine (L40S), un seul profil de robot, caméras 224×224 ; changer `camera_size` ou le robot change les coûts.
- Plusieurs serveurs réels (plusieurs GPU) n'ont pas été mesurés ; le client multi-serveurs est testé côté CPU avec des serveurs factices.
- Durées des tests : fidélité ≈ 20 min, coût (4 tailles) ≈ 11 min, un réglage du balayage de doigts ≈ 7 min ; un `salloc` interactif est tué à sa limite de temps (≈ 25 à 35 min observées), d'où `jobs/job_isaac_tests.sh` (4 h).

---

## 7. Reproductibilité : ce qui est garanti et ce qui ne l'est pas

Il y a **deux** propriétés différentes, souvent confondues :

1. *« La même séquence d'appels redonne les mêmes résultats. »* **Garanti.**
2. *« Le départ de `(seed, round)` est le même quel que soit ce qui a tourné avant. »* **Non garanti** avec le Robotiq.

### 7.1 Garanti

**G1. Même séquence, mêmes bits.** Rejouée deux fois depuis un serveur **fraîchement démarré** (même YAML, même machine), une même séquence de `reset` et de `step` donne des observations identiques bit à bit (différence 0,0 sur les 7 clés non-image, tactile compris ; les resets des rounds sont inclus). Test : `test_the_same_sequence_on_a_fresh_server_is_bit_exact` (4 environnements, 20 pas, 3 rounds). Le « plancher » de l'état complet (365 tenseurs : joints, liens, cheville, vitesses, accélérations…) pour deux historiques identiques est 0 partout. Conditions : `enhanced_determinism: true` et `CUBLAS_WORKSPACE_CONFIG=:4096:8` (posés par `jobs/env_server.sh` et par les tests).

*Portée.* Testé sur 20 pas ; `test_long_run_is_stable` vérifie sur ~330 pas que rien n'explose (valeurs finies, cheville qui ne s'envole pas) mais **pas** l'égalité bit à bit sur de longues séquences. Le mécanisme (PhysX GPU déterministe) est le même, mais ce n'est pas mesuré au-delà.

**G2. Indépendance des environnements d'un round.** Piloter violemment l'environnement 0 ne change **rien** (différence 0,0) aux autres environnements, sur joints, force, offset, tactile, effort de pince, cheville. Test : `test_the_environments_of_a_round_are_independent`.

**G3. Un reset propre jusqu'à la saisie.** Le reset physique est identique bit à bit jusqu'à la fermeture de la pince sur la cheville (pas physique ~36 à 42 de la première passe, sur 134 pas de reset) ; avant, plancher 0. Avec le **Franka**, un autre épisode avant le reset ne change rien (0 différence sur 361 tenseurs).

**G4. Départs valides et variés.** Chaque départ reste un départ FORGE valide ; deux `(seed, round)` différents donnent des départs différents (~2,6 cm d'écart d'offset entre deux rounds, test `test_start_states_are_valid_and_diverse`) ; le round suivant est bien la graine suivante (`test_next_round_is_the_next_seed`).

**G5. Sémantique des épisodes.** Rien perdu, rien périmé à la frontière entre rounds ; `final_obs` correct ; `terminated` et `truncated` séparés (`test_a_round_ends_and_the_next_one_starts_by_itself`, tests CPU du manager, du collecteur et du protocole).

### 7.2 Non garanti

**N1. Le départ d'un `(seed, round)` après un autre épisode (Robotiq uniquement).** Comparaison du départ de `(seed, round 0)` après un simple reset contre le même départ après 24 pas d'une autre graine, 4 environnements × 6 départs = 24 « départs d'environnement » (réglage par défaut) :

| Mesure | Valeur |
|---|---|
| départs qui diffèrent | 8 sur 24 (6 à 8 sur 24 selon les réglages essayés) |
| écart médian sur la cheville, quand il y a écart | 0,01 à 0,06 mm |
| pire écart observé | 5,2 mm (réglage par défaut) ; 1 à 11 mm selon les réglages |
| écart maximal sur les joints | jusqu'à 1,4e-2 rad |

Conséquences concrètes :
- le départ obtenu reste un départ FORGE valide ;
- deux évaluations du « même round » faites avec des historiques différents partent d'un état **presque** identique, pas strictement identique ; `run_fixed_rounds` fixe donc l'application `(seed, round) → départ` **à cet écart près** ;
- les trajectoires qui suivent un tel départ ne sont pas garanties identiques (contacts : l'écart peut grandir) ;
- un rejeu bit à bit n'est garanti que si **toute la séquence** (resets, pas, actions) est identique depuis un serveur neuf (G1), ce qui suppose, pour un entraînement, des actions identiques aussi.

**N2. Le départ dépend de `(seed, round, num_envs, i)`.** Changer `num_envs` change les départs ; le départ de l'environnement `i` d'un serveur à N environnements n'est pas celui d'un serveur à un seul environnement. Comparer des runs au même `num_envs`. (Avec plusieurs serveurs, la graine du serveur *k* est décalée de `k × SEED_STRIDE`, 1 000 003.)

**N3. Les images.** Le test de séquence exclut les images ; le rendu n'a pas été comparé bit à bit entre deux serveurs. Ne pas le supposer.

**N4. Autre matériel ou autre version.** Toutes les mesures : L40S, Isaac Sim 5.1, Isaac Lab 2.3.0, torch 2.7.0. Un autre GPU, un autre pilote ou une autre version peuvent changer les bits.

**N5. Toute modification de la séquence ou de la configuration.** Un `reset` ou un `step` de plus, un autre `reset_passes`, `enhanced_determinism`, un gain PD, un réglage de doigt : cela change la suite des états. Les opérations `debug` du serveur (`state_dump`, `trace_next_reset`, etc.) sont des outils de test : ne pas les appeler pendant une collecte de données.

**N6. Effet sur la science, non mesuré.** Un écart de l'ordre du centième de millimètre en médiane est très petit devant la dispersion des départs FORGE ; l'effet sur un taux de succès à des centaines d'épisodes **n'a pas été mesuré**.

### 7.3 Cause (localisée, pas corrigée)

- Le reset est identique jusqu'à la fermeture de la pince sur la cheville ; là, des différences de niveau flottant (1e-9 à 1e-6) sont amplifiées d'un facteur 100 à 1000 par pas pendant 3 à 4 pas, puis saturent vers 1e-4 ; la seconde passe de reset les efface dans la plupart des environnements (d'où `reset_passes: 2`).
- Il s'agit d'un **état caché dans le solveur de contacts de PhysX**, ni lisible ni réinitialisable depuis Python.
- Ce n'est pas : l'endormissement des corps (`disable_sleep` sans effet), les taxels (mêmes écarts avec ou sans zones tactiles dans l'ancien dépôt), l'amortissement ou la raideur des doigts, les paramètres de friction PhysX testés.
- La saisie du Robotiq fait saturer le moteur de doigt à 6,0 N·m pendant la prise : raideur et amortissement ne changent donc pas la force de serrage.

### 7.4 Ce qui a été essayé (`tests/test_isaac_finger_sweep.py`, 4 environnements × 6 départs)

| Essai | Résultat |
|---|---|
| `reset_passes` 2 au lieu de 1 | 1,5 → 0,2 mm sur une graine (mesure de l'ancien dépôt) ; une 3ᵉ passe n'apporte rien ; adopté |
| `disable_sleep` | aucun effet |
| `finger_damping` 20 / 100 / 500, `finger_stiffness` 2000 | toujours 6 à 8 départs sur 24 |
| `physx_friction_correlation_distance` 0,0005, `physx_friction_offset_threshold` 0,002 | toujours 6 à 8 sur 24 |
| `finger_effort` 3 | 7/24, pire écart 1,2 mm, mais force tactile divisée par ~2 et chevilles perdues : change la tâche, rejeté |
| `physx_solver_type` 0 (PGS) | casse la prise (cheville lâchée dans 18 cas sur 24) |
| `physx_friction_correlation_distance` 5e-5 | le serveur s'arrête (code 1), cause non investiguée |

Ce qui resterait : les articulations mimic ou les formes de collision de l'actif Robotiq, ou l'intérieur de PhysX. Jugé non rentable : cela obligerait à refaire démonstrations et SFT.

### 7.5 Recommandations d'usage

1. **Comparer des conditions sur la même séquence de rounds, depuis un serveur neuf, avec le même `num_envs`.**
2. Garder `enhanced_determinism`, `reset_passes` et tout le YAML identiques pour **toutes** les données d'une expérience ; consigner avec chaque run le YAML, le commit git et les versions d'Isaac Sim / Isaac Lab.
3. Pour une évaluation, `run_fixed_rounds` avec les mêmes `rounds` et la même `seed` ; formuler le résultat comme « mêmes `(seed, round)` », **pas** « mêmes états bit à bit ».
4. Si l'indépendance vis-à-vis de l'historique était indispensable (hors toucher), le profil `franka` l'a ; il n'a pas de taxels et ne répond donc pas à la question de recherche.
5. Ne pas appeler les opérations `debug` pendant une collecte.

---

## 8. Tests

| Commande | Ce qu'elle fait |
|---|---|
| `uv run pytest tests -q` | tests CPU (97 passés, 11 ignorés : ceux qui demandent Isaac) : protocole, client/serveur, manager, tracker, collecteur, coût, tests statiques d'architecture |
| `SLURM_SUBMIT_DIR=$PWD bash jobs/job_isaac_tests.sh tests/test_isaac_fidelity.py` | fidélité : ce qui survit à un reset, divergence du reset, dépendance à l'épisode précédent, départs valides, indépendance, long run, séquence bit à bit |
| `COST_NUM_ENVS=1,4,16,64 SLURM_SUBMIT_DIR=$PWD bash jobs/job_isaac_tests.sh tests/test_isaac_cost.py` | temps et VRAM en fonction de N (autres variables : `COST_STEPS`, `COST_RESETS`, `COST_MEASURED_EPISODE`, `COST_ROLLOUT_STEPS`, `COST_EPISODE_STEPS`) |
| `SLURM_SUBMIT_DIR=$PWD SWEEP="default;finger_damping=50" SWEEP_TAG=x bash jobs/job_isaac_tests.sh tests/test_isaac_finger_sweep.py` | balayage des réglages de doigt / PhysX contre le résidu (3 réglages par job, ≈ 7 min chacun) ; rapport `reports/finger_sweep_<tag>.json` |

Les tests Isaac acceptent `--isaac-config` et `--isaac-set KEY=VALUE` pour changer de YAML ou de réglage. Les tests statiques vérifient notamment qu'aucun fichier de `src/envs/` n'importe un simulateur ou torch, ni le sous-paquet `isaaclab`.

---

## 9. Ce qui n'a pas été mesuré

- N > 64 (et la VRAM / le temps réels à grand N) ; plusieurs serveurs sur plusieurs GPU avec Isaac.
- Égalité bit à bit sur de longues séquences (> 20 pas) et du rendu des images.
- Un autre GPU, pilote ou version d'Isaac Sim / Isaac Lab.
- L'effet du résidu de départ (N1) sur un taux de succès ou sur la comparaison de deux politiques.
- La cause du crash du serveur avec `physx_friction_correlation_distance=5e-5`.
- La perte / l'écrasement de cheville sur des centaines de graines pour les réglages du balayage (critère qui avait fixé `finger_effort = 6`).
