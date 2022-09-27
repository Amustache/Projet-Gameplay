# Projet Gameplay
README en cours de construction.

## Structure actuelle
* Le fichier requirements.txt contient pour l'instant un peu un fourre-tout des packages que l'on sera susceptible d'utiliser. Ne pas hésiter à le compléter.
* Le fichier .gitignore à la racine est configuré pour être le plus vaste possible, et éviter de push n'importe quoi de pas strictement utile.
* Le dossier temp est ignoré par défaut, il ne faut pas hésiter à l'utiliser pour faire du bricolage directement.
* Le reste, ce sont des fichiers de configuration que l'on peut totalement ignorer pour le moment - ce sera pour faire du joli code par la suite (;.

## Comment faire le Python
Déjà, il faut avoir [Python](https://www.python.org/downloads/) d'installé ! Également, il faut avoir [pip](https://pip.pypa.io/en/stable/installation/) d'installé !

Ensuite :
1. Créer un nouvel environnement de travail: `python -m venv ./env`
2. Activer l'environnement de travail: Linux: `source ./env/bin/activate` ; Windows: `.\env\Scripts\activate`
3. Installer les dépendances: `pip install -Ur requirements.txt`
