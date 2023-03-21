# Projet Gameplay
Le Projet Gameplay est un projet de semestre visant à étudier la gramaire de plusieurs jeux vidéo.

## Setup
### General
Ce projet nécessite Python 3.10+ et Pip 22.3+.

1. Cloner ce repository : `git clone git@github.com:Amustache/Projet-Gameplay.git`
2. Créer un nouvel environnement de travail : `python -m venv ./env`
3. Activer l'environnement de travail :
   * Linux : `source ./env/bin/activate`
   * Windows : `.\env\Scripts\activate`
4. Installer les dépendances : `pip install -Ur requirements.txt`
   * Sur Debian, installer les dépendances : ``
5. Activer pre-commit : `pre-commit install`

### ML
TODO

### Webapp
Dans le dossier `Webapp` :

1. Copier le template de configuration : `cp config.dist.py`
2. Éditer le fichier pour modifier `SECRET_KEY` avec quelque chose de secret.
3. Télécharger les [données d'exemple](#), et les placer dans le dossier `Webapp/static/inputs/demo/`.

## Utilisation
### General
TODO

### ML
Dans le dossier `ML`.

#### Prédiction
1. Préparer la vidéo (n'importe quel format utilisable, exemple .mp4).
2. Faire une prédiction : `python3 main.py predict <chemin/vers/la/video> model.pth`
3. Le fichier de sorti est `<le nom de la video>.csv`

#### Conversion vers JS
* `python3 utils.py csvToJs <chemin/vers/le/video.csv> <prediction_data>`

### Webapp
Dans le dossier `Webapp`, lancer l'application avec `python app.py`.
