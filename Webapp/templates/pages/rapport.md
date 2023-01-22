_Par souci de lisibilité, dans ce rapport on emploiera par défaut le féminin si le genre de la personne désignée n’importe pas._

## 1 Introduction
Le projet considéré consiste à extraire des données de jeu à partir de vidéos de _gameplay_ ("partie de jeu") d’un jeu vidéo afin d’en faire une analyse comparative entre joueuses, cela dans le but de classifier si possible les joueuses par leur façon d’aborder et jouer à un jeu donné.

Les deux jeux considérés sont "_Super Mario Bros._" [<a href="#Nin85">${ref}</a>] et "_The Legend of Zelda : _Breath of the Wild__" [<a href="#Nin17">${ref}</a>], abrégé _Breath of the Wild_ dans ce rapport. _Super Mario Bros._ est un jeu de plates-formes linéaire emblématique et relativement simple dans son gameplay, ce qui en fait un premier choix de qualité pour ce projet. _Breath of the Wild_ quant à lui est un jeu d’action-aventure en monde ouvert, choisi pour sa complexité permettant un plus grand défi et potentiellement une plus grand variété dans les résultats attendus.

Ces deux jeux étant de nature totalement différente, deux approches sont considérées : pour _Super Mario Bros._ les données de jeu examinées sont les touches pressées par la joueuse tandis que pour _Breath of the Wild_, étant donné son gameplay non-linéaire encourageant l’exploration, nous nous sommes concentrés sur les trajectoires de déplace- ment effectuées par la joueuse.

Mon travail devait initialement porter sur _Breath of the Wild_ avec l’aide de Hugo Hueber, et en particulier sur l’analyse des trajectoires extraites, tandis que Corentin Junod s’attelait à _Super Mario Bros._ Cependant nous avons sous-estimé la difficulté et le temps nécessaire à l’extraction des trajectoires, celle-ci s’est donc avérée plus longue que prévu et le projet a dû être remanié en cours de route, ce qui explique le manque d’analyse pour ce jeu et mon passage sur le cas d’étude _Super Mario Bros._. Par souci d’exhaustivité, nous parlerons tout de même des deux cas lorsque cela est applicable.

### 1.1 Questions de recherche et objectifs
#### 1.1.1 _Super Mario Bros._
Dans le cadre de _Super Mario Bros._, on souhaite initialement savoir si il est possible de classifier les différents styles de jeu en se basant uniquement sur les données entrées par la joueuse, c’est-à-dire les boutons pressés en cours de jeu.

Suite au remaniement du projet, on étudie finalement l’existence de boucles de jouabilité, qu’on définit comme étant des séquences d’action que la joueuse va répéter encore et encore au sein d’un jeu et qui peuvent être définies à différents niveaux : petites, moyennes ou grandes. Pour un jeu tel que _Pokémon_, on pourrait par exemple définir la boucle de jouabilité principale de la façon suivante : "Attraper des pokémons"	"Combattre" "Gagner des badges"	"Attraper des pokémons", où chaque action pourrait elle-même se diviser en boucles plus petites.

On étudie les boucles de jouabilité dans _Super Mario Bros._ par le biais des touches pressées. Le but est de savoir si de telles boucles existent et, le cas échéant, comment les repérer et les visualiser.

#### 1.1.2 _Breath of the Wild_
Pour _Breath of the Wild_, la question est double : peut-on classifier les différents types de joueuses à partir de leurs déplacements et le cas échéant, peut-on définir une grammaire de déplacement pour cette typologie d’expériences de jeu.
Une question sous-jacente à celle de l’existence est celle de la récurrence : est-ce qu’un certain type de déplacement est propre à un certain type d’expérience de jeu ou est-ce qu’il est commun à plusieurs ?

On définit la "grammaire de déplacement" comme étant l’ensemble des schémas de construction, c’est-à-dire l’ensemble des règles implicites et explicites, qui régissent les déplacements des joueuses sur la carte. Ceci inclut donc les types de déplacements (rectiligne, courbé, hasardeux...) en fonction des contraintes et influences auxquelles ces dépla- cements sont soumis (un objectif devant être atteint, une silhouette au loin attisant notre curiosité...). Cette grammaire de déplacement façonne donc l’expérience de jeu proposée par _Breath of the Wild_, qui peut être définie à différents niveaux de granularité : expérience "décontractée" ou "inconditionnelle" ("_casual gamer_" vs. "_hardcore gamer_") ou à un niveau plus fin, avec des expériences de type "chasseuse-cueilleuse", "exploratrice" ou encore "collectionneuse".

### 1.2 État de l’art
Le concept du projet est inspiré du travail de Mathieu Triclot dans son essai [<a href="#Tri19">${ref}</a>] visant à étudier la relation entre la joueuse et le jeu vidéo en enregistrant les entrées produites sur des manettes, sans utiliser les images affichées à l’écran. Alors que M.Triclot se penche plutôt du côté de la comparaison de jeux, on souhaite ici se rapprocher de la modélisation de type de joueuse et de gameplay.

La notion de grammaire de déplacement est inspirée par celle de grammaire de jeu, redéfinie par Guillaume Grandjean dans sa thèse [<a href="#Gra20">${ref}</a>], et avec laquelle Grandjean formalise et hiérarchise différentes structures spatiales et leur mise en place dans la série de jeux vidéos _The Legend of Zelda_.

Dans une série d’études [<a href="#WK12">${ref}</a>] [<a href="#Wal13">${ref}</a>], [<a href="#WK14">${ref}</a>], Walner et al. développent d’abord un concept de formalisation de données de jeu et une méthodologie d’analyse de ces données formalisées avant de développer un outil d’analyse visuelle pour explorer et com- prendre de grandes quantités de données comportementales de joueuses. Une autre étude [<a href="#DC09">${ref}</a>] introduit l’approche de l’analyse spatiale des jeux vidéos par le biais des systèmes d’information géographiques afin de compléter les méthodes de test/recherche actuelles qui sont centrées sur l’utilisatrice.

## 2 Présentation des données
Dans les deux cas la donnée de travail initiale est une vidéo de gameplay, c’est-à-dire l’enregistrement vidéo d’une partie effectuée par une joueuse. A cet effet, un corpus de vidéos provenant majoritairement de la plate-forme d’hébergement YouTube est constitué, avec l’addition de diverses données qualitatives et méta-données si disponibles. La constitution du corpus et sa revue ont été réalisées avec l’aide de Magalie Vetter, assistante scientifique au Collège des Humanités.

### 2.1 _Super Mario Bros._
#### 2.1.1 Constitution du corpus d’étude
On recueille les vidéos avec leur date de publication, la chaîne de publication, le type de gameplay, si la partie est complète ou non et divers défis imposés par la joueuse (pas de mort, pas de raccourci, toutes les pièces...). On notera également qu’à des fins d’en- traînement, nous avons enregistré des parties jouées par des étudiantes sur le campus de l’EPFL (voir section 2.1.2).

#### 2.1.2 Transformation en données exploitables
On souhaite effectuer le travail d’analyse sur les trois actions suivantes : déplacement à gauche, déplacement à droite et saut. Pour ceci nous avons besoin de reconnaître quelle touche a été pressée à un instant donné de la vidéo. A cette fin, Corentin a développé un modèle de machine learning qui à partir d’une vidéo de gameplay, prédit quelles touches ont été effectivement pressées ou relâchées pendant la partie.

Ce modèle est passé par plusieurs itérations et celle utilisée pour l’obtention des données à analyser fonctionne de la façon suivante : il s’agit d’un réseau de neurones à convolution qui à un instant t prend en entrée une image de la vidéo et la traite, et de même pour un certain nombre N d’images suivantes, où N dépend du jeu. Il faut par exemple prendre beaucoup d’images supplémentaires pour _Super Mario Bros._ car l’avatar à l’écran a beaucoup d’inertie (il "glisse") ce qui implique qu’un déplacement à l’écran ne signifie pas forcément qu’une touche est pressée. Ensuite, les sorties sont assemblées en une représentation intermédiaire apprise par le modèle où l’on spécifie seulement le nombre de paramètres produits en sortie. Ces paramètres sont ensuite utilisés dans trois blocs de logique identiques séparés, un par bouton qu’il faut prédire.

Avec ce système et avec les données récoltées sur l’EPFL, on obtient des prédictions correctes pour 80 à 90% des images de la vidéo si on considère les boutons de façon individuelle, et entre 70 et 80% si on considère les 3 boutons simultanément.

Afin d’avoir des données d’entraînement pour le modèle et pour commencer les premières analyses, des ordinateurs permettant de jouer à _Super Mario Bros._ sur émulateur ont été mis à disposition des étudiantes de la Faculté IC pendant une semaine. Pendant les sessions de jeu, nous avons recueilli grâce au logiciel OBS l’enregistrement vidéo de la partie ainsi que les touches pressées. Les sessions ont duré entre 5 et 20 minutes. Nous avons aussi recueilli l’expérience avant/après des joueuses avec le jeu ainsi qu’avec les jeux vidéos en général.

![fig1](fig1.png)
Figure 1 – Diagramme du modèle

Une limitation de ces données d’entraînement est que les parties enregistrées sont de fait souvent restreinte au premier niveau car la prise en main du jeu est plus difficile que prévue.

A ce stade, les données à exploiter sont des tableaux où chaque ligne correspond à une touche, son état (pressée/relâchée) et l’image à laquelle cela se produit, avec autant de lignes qu’il y a de touches pressées et relâchées durant une partie.

### 2.2 _Breath of the Wild_
#### 2.2.1 Constitution du corpus d’étude
Pour ce jeu, un corpus de départ de 100 parties a été établi. Le jeu étant relativement long et complexe, les données sont très hétérogènes et un grand nombre de méta-données ont été ajoutées afin de faciliter le traitement et l’analyse des données plus tard dans la pipeline. Ce corpus contient en outre :
* le lien vers la vidéo contenant la fin du didacticiel du jeu, moment choisi pour débuter l’extraction et l’analyse des trajectoires
* l’horodatage de la fin du didacticiel
* le nombre de vidéos pour une partie de jeu (allant de 1 à parfois plus de 300)
* la date de publication de la première vidéo
* le type de gameplay ("_casual_", "_100%_", "_speedrun_"...)
* si la run est complète ou non
* si tous les sanctuaires ont été complétés (objectif secondaire classique du jeu)
* si du contenu additionnel ("_DLC_") est utilisé
* la chaîne de publication
* si les vidéos sont dans une liste de lecture YouTube et le cas échéant, un lien vers celle-ci
* si la vidéo est en plein écran
* si il y a du bruit sur la mini-carte
* si la partie est jouée en mode "normal" ou "difficile" ("_Master Mode_")
* si la liste de lecture contient seulement du gameplay ou aussi des vidéos diverses
* si c’est la première partie effectuée par la joueuse
* l’expérience de la joueuse avec les jeux-vidéos en général
* l’objectif de la joueuse avec la publication de ces vidéos ("_divertissement_", "_découverte du jeu_", "_soluce_"...)
* le contexte d’enregistrement de la vidéo ("_uploader_" vs. "_streamer_")

![fi2](fig.jpg)
Figure 2 – Exemples d’entrée du corpus

#### 2.2.2 Transformation en données exploitables
L’idée de base est la suivante : à partir d’une vidéo de gameplay, on extrait la mini-carte située à l’écran contenant en son centre une flèche représentant la position et l’orientation de l’avatar, ainsi que diverses informations de jeu dont en particulier la position du nord et les courbes de niveau du terrain. Puis, étant donné que cette mini-carte est une fraction identique de la carte globale disponible dans les menus du jeu, on reconstitue la position et donc la trajectoire de déplacement de la joueuse en faisant correspondre les éléments présents sur la mini-carte avec ceux de la carte globale.

Très rapidement plusieurs problèmes se posent :
1. la quantité de données à stocker et exploiter : les parties durent en moyenne une centaine d’heure, et avec une qualité d’image assez haute pour effectuer le traitement on atteint plusieurs téraoctets de données, ce qui n’est pas viable
2. la mini-carte n’est pas toujours visible et peut contenir du bruit externe au jeu, comme par exemple l’espace de discussion des spectateurs si l’enregistrement pro- vient d’un enregistrement en direct
3. la carte n’est pas toujours découverte : en effet, il faut d’aborder découvrir la zone en effectuant l’ascension d’une tour, faute de quoi la carte et la mini-carte sont simplement composées de la flèche de l’avatar et d’un quadrillage bleu clair sur fond bleu foncé se répétant à l’infini (figure 3)
4. la joueuse peut décider de se téléporter à certains endroits, et en particulier à des endroits pour lesquels la carte n’est pas encore découverte
5. l’image de la mini-carte une fois extraite est de basse résolution
6. la mini-carte n’est pas opaque et donc certains éléments du décor du jeu sont visibles à travers la mini-carte (figure 3b)

Pour les deux premiers problèmes, il a été décidé de garder uniquement les vidéos qui semblent ne pas avoir de bruit sur la mini-carte et de masquer la vidéo afin de ne garder que la mini-carte à traiter. Une fois la mini-carte traitée, on ne garde qu’un tableau contenant les coordonnées de la position de l’avatar à intervalles réguliers.

Pour ce qui est de la carte non-découverte, l’idée est de se servir du quadrillage pour reconstruire la trajectoire. Si l’avatar passe d’un endroit connu à inconnu alors on a un point de repère de départ exploitable. Si on arrive en zone inconnue suite à une téléportation alors on garde en mémoire la trajectoire jusqu’à atteindre un point de repère connu, par exemple une portion de carte dévoilé, puis on effectue le chemin inverse à partir de ce point pour reconstruire la trajectoire.

![fig3a](fig3a.png)![fig3b](fig3b.png)

Figure 3 – Exemples de mini-carte pouvant apparaître à l’écran, montrant clairement la transparence lorsque non-découverte. (a) Découverte	(b) Non-découverte

On commence avec le cas de la carte non-découverte. On extrait l’image de la mini- carte à un instant t ainsi qu’à un instant t + δ puis, à l’aide de la librairie OpenCV, on effectue une mise en correspondance des éléments des deux images (figure 4) afin de pou- voir calculer ensuite la rotation et la translation qu’il s’est produit entre ces deux instants et en déduire le déplacement (figure 5).

![fig4](fig4.jpg)
Figure 4 – Mise en évidence des éléments communs entre deux images (presque) successives de la mini-carte

Malheureusement, bien que les résultats passent approximativement le test de l’oeil, ils ne sont pas assez précis sur une période de quelques secondes, ce qui impliquerait des écarts énormes sur des périodes de plusieurs heures. De plus, des "sauts" inexplicables apparaissent à divers moments et nous supposons que ceci est dû à des approximations lors du calcul des rotations. A ce stade il est clair que nous avons sous-estimé la tâche de l’extraction des trajectoires et que nous n’aurons pas de données exploitables à temps pour en faire l’analyse. Le projet _Breath of the Wild_ est donc abandonné et Hugo et moi-même nous reportons sur le projet _Super Mario Bros._.

![fig5](fig5.jpg)
Figure 5 – Report sur un plan de la trajectoire estimée

## 3 Analyse des données
A partir de cette section, seulement le cas de _Super Mario Bros._ sera abordé. On souhaite commencer par trouver la boucle de jouabilité minimale, c’est-à-dire la plus petite séquence de touche qui se répète indéfiniment.

### 3.1 Visualisations   préliminaires
On propose deux types de visualisation pour se familiariser avec les séquences d’ac- tions effectuées.

La première figure (figure 6), statique et réalisé en Python avec Matplotlib, ne considère que les touches pressées. Chaque trait vertical représente la pression d’une touche et l’axe des abscisses correspond à la position de la touche pressée par rapport à la totalité des touches pressées durant une partie. On constate la prédominance du saut et du déplacement à droite pour cette partie donnée.

La deuxième figure (figure 7), dynamique et réalisé en Javascript considère de plus le relâchement des touches. Un rectangle débute quand la touche est pressée et se termine quand elle est relâchée. On voit dans ce cas que la joueuse a effectué plusieurs sauts successifs tout en courant à droite. Cette visualisation est prévue pour une page web. En l’état, l’axe des abscisses correspond aussi à la position de la touche mais il est prévu de remplacer cet axe par un axe temporel, où le défilement est synchronisé avec la lecture d’une vidéo de gameplay.

### 3.2 Détection de motif dans une chaîne de caractères
Afin de détecter les boucles de jouabilité, la première idée est de réduire le problème à la détection de motif au sein d’une chaîne de caractères.

![fig6](fig6.png)
Figure 6 – Visualisation "code barre" des touches pressées

![fig6](fig7.png)
Figure 7 – Visualisation dynamique des actions simultanées

#### 3.2.1 Préparation
On convertit chaque ligne du tableau en lettre où chaque caractère correspond à une touche pressée/relâchée, on note "r" quand la touche droite est pressée et "R" quand elle est relâchée, en faisant de même pour la touche gauche et saut ("L" et "J"). Ainsi, une séquence d’action où la joueuse court à droite, saute en courant puis s’arrête correspond à la chaîne de caractère "rjJR". Une fois le tableau converti, on obtient une chaîne de caractère représentant la partie entière.

#### 3.2.2 Première approche : force brute
On s’intéresse d’abord seulement aux touches pressées, divisant donc la longueur de la chaîne par deux. La première approche est une approche de force brute qui consiste à calculer toutes les permutations possible de tailles 2 à N , à les stocker dans un dictionnaire et à parcourir la chaîne de caractère en incrémentant de 1 la valeur associée au motif si celui-ci est détecté lors du parcours de la chaîne.

Comme on peut s’y attendre, les temps de calculs deviennent exponentiellement longs pour N grand, en particulier à partir de N = 6. Pour N 4 on obtient cependant déjà un premier résultat intéressant : les séquences d’action de taille 2 les plus fréquentes semblent être "droite, saut" et "saut, droite" tandis que pour les tailles 3 et 4 il s’agit de "saut, droite, saut" et "droite, saut, saut, saut".

On remarque d’ailleurs deux autres problèmes : en ne s’intéressant qu’aux touches pressées, on perd une partie des séquences d’intérêt possibles, par exemple lorsque la joueuse saute plusieurs fois sans s’arrêter de courir. De plus, en analysant une partie dans sa totalité on perd l’information de la fréquence d’apparition, autrement dit, est-ce que les séquences les plus présentes le sont parce qu’elles apparaissent à intervalles réguliers ou parce qu’un environnement de jeu spécifique invite à répéter une même séquence un grand nombre de fois dans un laps de temps restreint.

#### 3.2.3 Deuxième approche : découpage de partie en segments uniformes
On propose plusieurs améliorations : on ne pré-calcule plus la totalité des permutations de touches possibles mais on considère seulement celles qui apparaissent effectivement dans la chaîne de caractère, de plus si un motif de taille k n’apparaît pas alors un motif de plus grande taille ne peut pas contenir ce motif, ce qui permet de réduire considérable- ment le nombre de motifs à considérer. On ajoute également l’implémentation en Python d’un algorithme de détection de motif plus rapide : l’algorithme de Knuth–Morris–Pratt [<a href="#KMP77">${ref}</a>] qui permet aussi de garder en mémoire la position des motifs. Enfin on inclut le relâchement des touches et on découpe une partie en segments uniformes. Ce découpage sert à détecter si il y a un laps de temps pour lequel une séquence d’action apparaît significativement plus par rapport aux autres. On découpe la partie de jeu en segments identiques allant de 3 à 10 secondes, avec pas de 0.5s et on calcule les fréquences d’apparitions des motifs de taille 4 à 8 (soit entre 2 et 4 touches pressées et relâchées).

On constate plusieurs choses : peu importe la longueur du segment et la taille des motifs, le seul motif prédominant est la suite de sauts, qui apparaît avec une fréquence en moyenne 2 fois plus élevée que le deuxième motif le plus fréquent. Il n’y a pas de différence significative pour différentes longueurs de segment et nous laissons donc de côté l’idée de découper une partie en plusieurs segments. A ce stade, on n’observe donc pas de particularité significative.

### 3.3 Influence d’une séquence d’action sur la suivante
Cette méthode cherche à savoir si une touche pressée à un certain endroit du motif influe la prochaine touche pressée. Pour cela on considère de nouveau seulement la pression des touches et on construit un graphe de la façon suivante : pour une taille de motif N , on crée 3 lignes de N noeuds, chaque ligne correspondant à une action (de haut en bas : gauche, saut, droite) et chaque colonne à une position dans le motif. Un segment ne peut que relier 2 noeuds qui sont dans des colonnes successives et chaque segment possède un poids qui correspond au nombre de fois où deux actions se suivent à une position donnée du motif. Ainsi plus le poids d’un segment est grand, plus la succession des touches à cet endroit est présente et plus le segment sera épais. (figure 8)

On observe dans ce cas que pour une séquence de 4 actions, cette personne va la majorité du temps faire suivre un saut d’un autre saut ou d’un mouvement à droite sans différence significative, tandis qu’elle va avoir tendance à faire suivre un mouvement à droite par un saut.

![fig8](fig8.jpg)
Figure 8 – Exemple de graphe pour une partie donnée et des motifs de taille 4

## 4 Discussion
### 4.1 Limitations
La majorité des résultats présentés dans ce rapport sont tirés d’analyses sur des données d’entraînement et ne sont donc pas représentatifs du corpus d’étude constitué. De plus, en restreignant le corpus d’étude initial à des vidéos misent en ligne par des personnes de leur plein gré, on inclut probablement un biais de sélection impliquant que les résultats obtenus ne sont valables que pour une certaine population des joueuses ne représentant pas forcément la joueuse moyenne.

Il est donc tout à fait possible qu’en considérant des données couvrant un éventail de type de joueuses plus large, nous obtiendrions des résultats totalement différents.

### 4.2 Travaux Futurs
Maintenant que le modèle de prédiction des touches donne des résultats satisfaisants sur les données d’entraînement, il serait intéressant d’étendre ces analyses aux vidéos du corpus sur lesquelles on aura appliqué le modèle de prédiction, afin de vérifier les hypothèses émises par l’analyse des données d’entraînement. Il serait d’autant plus intéressant de pouvoir sélectionner des joueuses au hasard pour augmenter la représentativité des données et éviter tout biais de sélection.

Une autre approche à explorer pour tenter de détecter les boucles de jouabilités, proposée par Corentin, est l’utilisation de l’algorithme de compression _Byte Pair Encoding_ [<a href="#Gag94">${ref}</a>] sur l’ensemble de la partie. Pour ceci, on considère les combinaisons possibles de touches pressées comme des états. Il y a au départ 8 états possibles, allant de "aucune touche n’est pressée" à "les trois touches sont pressées simultanément". Puis, on cherche la paire d’état la plus fréquente et cette paire devient un nouvel état à part entière. On cherche de nouveau la paire d’état la plus fréquente en considérant de plus l’état précédemment créé. On continue ainsi jusqu’à ne plus avoir aucune paire d’état qui apparaît plus qu’une autre. En analysant ces séquences d’état, il est potentiellement possible de retrouver les boucles de jouabilités.

Un axe d’analyse différent de ce qui est proposé jusqu’à maintenant serait "d’opérationnaliser" la notion de boucle de jouabilité. Développée par Franco Moretti [<a href="#Mor05">${ref}</a>], la notion d’opérationnalisation est une approche de l’analyse littéraire visant à utiliser des méthodes quantitatives pour étudier la littérature de manière systématique et rigoureuse. Plus précisément, cela consiste à définir les concepts, ici celui de la boucle de jouabilité, de manière à ce qu’ils puissent être mesurés de manière objective, afin d’en faire des analyses quantitatives. De cette façon, il serait possible de comparer les boucles de jouabilité d’un jeu d’un point de vue dit "intuitif", c’est-à-dire comme définie dans la section 1.1.1, mais aussi d’un point de vue dit "empirique", grâce à l’opérationnalisation de la notion.

## 5 Références
* <strong id="KMP77">[KMP77]</strong> Donald E. Knuth, James H. Morris Jr. et Vaughan R. Pratt. “Fast Pattern Matching in Strings”. In : SIAM Journal on Computing 6.2 (1977), p. 323-350. doi : 10.1137/0206024. eprint : https://doi.org/10.1137/0206024. url : https://doi.org/10.1137/0206024.
* <strong id="Nin85">[Nin85]</strong>      Nintendo. _Super Mario Bros._ Jeu. 1985.
* <strong id="Gag94">[Gag94]</strong> Philip Gage. “A New Algorithm for Data Compression”. In : C Users J. 12.2 (fév. 1994), p. 23-38. issn : 0898-9788.
* <strong id="Mor05">[Mor05]</strong>  Franco Moretti. Graphs, Maps, Trees : Abstract Models for a Literary His- tory. Verso, 2005.
* <strong id="DC09">[DC09]</strong> Anders Drachen et Alessandro Canossa. “Analyzing spatial user behavior in computer games using geographic information systems”. In : Proceedings of the 13th International MindTrek Conference : Everyday Life in the Ubiquitous Era on - MindTrek ’09. the 13th International MindTrek Conference : Everyday Life in the Ubiquitous Era. Tampere, Finland : ACM Press, 2009, p. 182. isbn : 978-1-60558-633-5. doi : 10.1145/1621841.1621875. url : http://portal.
* acm.org/citation.cfm?doid=1621841.1621875 (visité le 28/09/2022).
* <strong id="WK12">[WK12]</strong> Günter Wallner et Simone Kriglstein. “A spatiotemporal visualization approach for the analysis of gameplay data”. In : Proceedings of the SIGCHI Conference on Human Factors in Computing Systems. CHI ’12 : CHI Confe- rence on Human Factors in Computing Systems. Austin Texas USA : ACM, 5 mai 2012, p. 1115-1124. isbn : 9781450310154. doi : 10.1145/2207676.
* 2208558. url : https://dl.acm.org/doi/10.1145/2207676.2208558
* (visité le 21/09/2022).
* <strong id="Wal13">[Wal13]</strong>    Günter Wallner. “Play-Graph : A Methodology and Visualization Approach for the Analysis of Gameplay Data”. In : (2013). url : http://www.fdg2013. org/program/papers/paper33_wallner.pdf.
* <strong id="WK14">[WK14]</strong> G. Wallner et S. Kriglstein. “PLATO : A visual analytics system for gameplay data”. In : Computers & Graphics 38 (fév. 2014), p. 341-356. issn : 00978493. doi : 10.1016/j.cag.2013.11.010. url : https://linkinghub. elsevier.com/retrieve/pii/S0097849313001891 (visité le 21/09/2022).
*  
* <strong id="Nin17">[Nin17]</strong>      Nintendo. The Legend Of Zelda : _Breath of the Wild_. Jeu. 2017.
* <strong id="Tri19">[Tri19]</strong> Mathieu Triclot. “Les jeux vidéo en aveugle : essai de rythmanalyses.” In : L’enquête et ses graphies en sciences sociales : figurations iconographiques d’après société. Amalion, 2019, p. 175-194.
* <strong id="Gra20">[Gra20]</strong> Guillaume Grandjean. “Le langage du level design. Analyse communica- tionnelle des structures et instances de médiation spatiales dans la série The Legend of Zelda (1986-2017).” Thèse de doct. Université de Lorraine, Centre de recherche sur les médiations (CREM), 2020. url : https://hal.univ- lorraine.fr/tel-03098076/document.