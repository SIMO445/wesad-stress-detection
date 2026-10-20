# Détection du stress à partir de signaux physiologiques multimodaux (WESAD)

> Projet individuel réalisé en remplacement du stage (été 2026)
> Baccalauréat en informatique, cheminement sciences des données
> Université du Québec à Trois-Rivières (UQTR), cours INF1036 Sujets spéciaux en TI

Ce projet combine **apprentissage automatique**, **apprentissage profond** et **inférence causale** sur des signaux
physiologiques enregistrés par des capteurs portables. Il distingue clairement deux questions :

1. **Prédiction** : peut-on reconnaître l'état d'une personne (état de base, stress ou amusement) à partir de ses signaux ?
2. **Causalité** : le fait d'être en condition de stress **cause-t-il** un changement de l'activité électrodermale (EDA)
   dans la fenêtre de temps suivante ?

## Jeu de données

**WESAD** (*Wearable Stress and Affect Detection*), Schmidt et al., 2018 : 15 sujets portant des capteurs à la poitrine
(RespiBAN) et au poignet (Empatica E4). Signaux : ECG, EDA, respiration, température, accélération, BVP.

Les données **ne sont pas incluses** dans ce dépôt (licence réservée à un usage académique). Pour reproduire le projet :

1. Téléchargez `WESAD.zip` depuis la page officielle : https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html
   (aussi référencé sur le dépôt UCI : https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection).
2. Décompressez-le pour obtenir un dossier `WESAD/` placé **à côté du notebook**.

## Démarche

```
Signaux bruts multifréquences
   -> fenêtres de 10 s non chevauchantes (moyenne, écart-type, min, max, pente par modalité)
   -> étiquette majoritaire par fenêtre (baseline, stress, amusement)
   -> modèles prédictifs (séparation par sujet entier)
   -> question causale (DAG, régression ajustée, forêt causale)
```

- **Prétraitement** : agrégation par fenêtres de 10 secondes, ce qui résout le problème des fréquences d'échantillonnage
  différentes (700 Hz pour l'ECG, 4 Hz pour l'EDA au poignet). Résultat : **3 320 fenêtres**.
- **Modèles prédictifs** :
  - Régression logistique
  - XGBoost (300 arbres)
  - TCN (*Temporal Convolutional Network*) implémenté avec PyTorch : 3 blocs de convolutions causales dilatées
  - ROCKET (10 000 noyaux de convolution aléatoires) suivi d'un `RidgeClassifierCV`
- **Évaluation rigoureuse** :
  - Séparation **par sujet entier** : 9 sujets d'entraînement, 3 de validation, 3 de test, sans aucun sujet partagé.
  - Validation croisée **GroupKFold** (5 plis regroupés par sujet).
  - Standardisation apprise uniquement sur l'entraînement, aucune variable future utilisée comme prédicteur.
  - Arrêt précoce du TCN sur un ensemble de validation distinct ; le test n'est consulté qu'une seule fois.
- **Inférence causale** :
  - Traitement : condition de stress. Sortie : EDA moyenne de la fenêtre suivante.
  - Ajustement : EDA de la fenêtre précédente, mouvement de la fenêtre précédente, temps écoulé.
  - Graphe causal (DAG), diagnostic de positivité et restriction au support commun.
  - Deux estimateurs : régression linéaire ajustée et **CausalForestDML** (EconML).

## Résultats

### Prédiction (650 fenêtres de test, 3 sujets jamais vus)

| Modèle | Exactitude | F1 macro |
|---|---|---|
| **Régression logistique** | **0,723** | **0,620** |
| XGBoost | 0,634 | 0,608 |
| ROCKET + RidgeClassifierCV | 0,680 | 0,586 |
| TCN (apprentissage profond) | 0,478 | 0,335 |

- Le modèle le plus simple généralise le mieux à de nouvelles personnes.
- Le TCN sur-apprend rapidement avec seulement 9 sujets d'entraînement. ROCKET, qui reçoit aussi la séquence brute
  mais n'apprend qu'un classifieur linéaire, confirme que la limite vient du nombre de sujets et non de la nature
  temporelle du signal.
- La condition de stress est la mieux reconnue ; l'état de base et l'amusement sont plus souvent confondus.

### Causalité (support commun : 1 106 fenêtres)

| Méthode | Effet estimé du stress sur l'EDA future | IC 95 % |
|---|---|---|
| Régression linéaire ajustée | +0,0537 µS | [0,014 ; 0,094] |
| CausalForestDML | −0,0134 µS | [−0,758 ; 0,732] |

La régression détecte un effet positif modeste, mais la forêt causale révèle une **forte hétérogénéité entre sujets**,
portée surtout par un sujet atypique. Ce désaccord montre pourquoi il ne faut jamais se fier à une seule méthode
d'estimation causale, et qu'un bon modèle prédictif ne répond pas à une question causale.

## Contenu du dépôt

```
.
├── wesad_project_final.ipynb   # Notebook complet et exécutable (code, figures, interprétations)
├── Rapport_Projet_WESAD.pdf    # Rapport détaillé du projet
├── requirements.txt            # Dépendances Python
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
jupyter notebook wesad_project_final.ipynb
```

Placez le dossier `WESAD/` à côté du notebook, puis exécutez toutes les cellules dans l'ordre.

## Technologies

Python, Pandas, NumPy, SciPy, Scikit-learn, XGBoost, PyTorch, sktime (ROCKET), EconML (CausalForestDML),
statsmodels, NetworkX, Matplotlib, Jupyter

## Auteur

**Mohamed Chouraki**, étudiant au baccalauréat en informatique (sciences des données), UQTR
GitHub : [SIMO445](https://github.com/SIMO445) · LinkedIn : [mohamed-chouraki](https://www.linkedin.com/in/mohamed-chouraki)

## Référence

Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018).
*Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.*
Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI '18).
