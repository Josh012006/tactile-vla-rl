# Démonstrations (SFT) : collecte, téléopération, vérification, conversion

Les démonstrations de FORGE PegInsert sont les **mêmes que celles d'expo-ft** : `demos` est un lien vers `/network/scratch/m/monganj/expo-ft/demos`, donc les dossiers existants se lisent tels quels, avec la même arborescence :

| Dossier | Contenu |
|---|---|
| `demos/isaaclab/ForgePegInsert_Robotiq_scripted` | démos scriptées propres |
| `demos/isaaclab/ForgePegInsert_Robotiq_noisy` | démos scriptées dont 35 % avec injection de bruit (étiquettes propres) |
| `demos/isaaclab/ForgePegInsert_Robotiq_teleop` | démos téléopérées |
| `demos/isaaclab/lerobot/expo_ft/forge_peg_insert_robotiq_*` | les mêmes, au format LeRobot (ce que lit openpi pour le SFT) |

Un `ep_NNNN_seedS.npz` contient `exterior_image`, `wrist_image`, `joint_position`, `gripper_position`, `ft_force`, `peg_socket_offset`, `actions` (7 cibles articulaires absolues + commande de pince), `prompt`, `seed`, et `tactile` (T, 2, 7, 4) et `gripper_effort` quand le serveur avait un robot à taxels. Vérifier ce que contiennent les démos existantes : `python -c "import numpy as np; print(np.load('<fichier>.npz').files)"`.

## Le serveur doit être réglé pour ces outils

Un seul environnement, un épisode qui se termine après 5 pas « inserted » consécutifs (la règle des démos : le script arrête l'enregistrement au même pas), pince fermée sur la cheville :

```bash
bash jobs/env_server.sh --set num_envs=1 --set success_hold_steps=5 --set gripper_can_open=false
```

Les scripts le vérifient au démarrage (`scripts/demos/single_env.py`) et s'arrêtent avec la commande à lancer si le serveur est réglé autrement. Les jobs ci-dessous le font eux-mêmes.

## Commandes

| Quoi | Commande |
|---|---|
| démos scriptées | `sbatch jobs/job_demos.sh collect --out demos/isaaclab/ForgePegInsert_Robotiq_scripted --num-episodes 50` |
| démos bruitées | `sbatch jobs/job_demos.sh collect --out demos/isaaclab/ForgePegInsert_Robotiq_noisy --num-episodes 50 --noise-fraction 0.35 --action-noise-sigma 0.06` |
| vidéos de la collecte | ajouter `--video-dir reports/demo_videos` (exterieur, poignet, panneau des taxels, bandeau) |
| rejouer des démos, avec vidéos | `sbatch jobs/job_demos.sh replay --demos-dir <dossier> --save-videos --video-first 20` |
| robot : images par phase et vidéo des taxels | `sbatch jobs/job_demos.sh robot_check --out reports/robot_check_tsf` |
| téléopération | `sbatch jobs/job_teleop_server.sh`, puis les deux commandes qu'il affiche (tunnel, `teleop_client.py` sur le PC) |
| conversion LeRobot | `python -m scripts.demos.convert_waypoint_demos_to_lerobot --demos-dir <dossier> --output-dir demos/isaaclab/lerobot --repo-name expo_ft/<nom>` (env `.venv`, besoin de `lerobot`) |
| statistiques de normalisation | `python -m scripts.demos.compute_demo_norm_stats --demos-dir <dossier> --output-dir assets/forge_demo_norm --asset-id <nom>` |
| analyses sans GPU | `scripts.demos.noise_effect`, `analyze_waypoint_failure`, `demo_z_bands` |

Les vidéos, la téléopération et `robot_check` affichent l'information tactile : deux grilles 7×4 (doigt gauche, doigt droit, ligne 0 = bout du doigt en bas), échelle de couleur 0 à 5 N, verdict (les deux doigts, un seul, aucun contact), maximum et somme des zones, nombre de zones en contact, force du poignet, effort de la pince. Le dessin est défini une seule fois dans `src/eval/video_panels.py`.

## Ce qui a changé par rapport à expo-ft

- Les scripts parlent au nouveau serveur (`VectorEnvClient`), pas à l'ancien `EnvClient`. `reset(seed)` est le tour 0 de cette graine ; chaque pas est envoyé avec `auto_reset=False`, donc un épisode terminé n'est jamais relancé derrière le dos du script (l'ancien code devait récupérer la dernière observation avant la remise à zéro).
- Ops serveur portées : `ik_joint_target` (IK de FORGE, tous les environnements), `finger_width`, `finger_distance`, `teleop_step` (IK + pas en un aller-retour, pour la téléopération à travers le tunnel) et `save_teleop_demo` (le serveur écrit le fichier, via le lien `demos`, sans jamais écrire hors du projet).
- La vérification du serveur (`num_envs`, `success_hold_steps`, `gripper_can_open`) vient de la spec du serveur, plus d'un YAML côté client.

## Pas repris

`grasp_check`, `grip_check`, `grip_stats` (calibration ponctuelle de la prise Robotiq), `repro_check`, `reset_history` (remplacés par `tests/test_isaac_fidelity.py`), `smoke_test_env_server` (remplacé par les tests), `analyze_peg_metrics` (lit les sorties de l'ancienne évaluation), `capture_forge_cameras` (construit son propre environnement Isaac : les caméras sont maintenant définies dans `src/envs/isaaclab/forge_env.py`) et les outils d'asset USD (`robot_assets.py` est déjà dans `src/envs/isaaclab`).

## À valider sur le cluster

Les ops qui lisent le simulateur (`ik_joint_target`, `finger_width`, `finger_distance`) ne se testent qu'avec Isaac : lancer d'abord `sbatch jobs/job_demos.sh collect --out demos/isaaclab/test_collect --num-episodes 2 --video-dir reports/demo_videos`. Les démos d'expo-ft ont été enregistrées avec l'ancien serveur : un `replay` dans le nouvel environnement mesure si la même graine donne le même départ (colonne « reset dev » et dérive en mm).
