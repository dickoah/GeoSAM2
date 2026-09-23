# TODO

## Bake / split : la texture comme canal d'étiquettes

État (branche `features/split`) : le bake garde les UV du maillage quand chaque
pièce y a de la place (`_atlas_usable`, ≥ 4 texels par face, pièces ≥ 50 faces),
sinon il régénère un atlas avec xatlas. Sélecteur `Atlas` (auto / keep / xatlas)
à l'étape 5. Les cinq références gardent leurs UV et leur bake est identique
octet pour octet ; le bureau (UV qui répètent un motif bois) est régénéré.

C'est la solution prudente sous la contrainte 1:1, pas la plus simple ni la plus
robuste.

### Le vrai problème est en amont

Le split lit les étiquettes à travers une texture. Sur un maillage dense, la
moitié des faces occupent moins d'un texel, même en 4096², quel que soit l'atlas.

| cas    | faces   | aire UV médiane / face (texels) | faces mal peintes, UV d'origine | faces mal peintes, xatlas |
|--------|---------|---------------------------------|---------------------------------|---------------------------|
| real_b | 187 032 | ≈ 0,7                           | 0,6 %                           | 2,3 %                     |
| real_c | 23 918  | 0,6                             | 3,6 %                           | 9,9 %                     |

Le canal « labels → texture → relecture » perd des faces par construction.

### Options, de la plus robuste à la plus conservatrice

1. **Split sur les étiquettes par face** (on les a déjà ; la texture ne sert plus
   qu'à l'affichage). L'atlas ne compte plus. Touche le split : à décider.
2. **Toujours xatlas.** Zéro heuristique, zéro mode, 0 à 2 s. Mais les bakes des
   références changent d'octets et le split de real_c change (10 pièces au lieu
   de 11, surface correcte 91,2 % au lieu de 86,2 %) : mieux, mais pas 1:1. Et
   xatlas ne peint pas mieux les petites faces, il les tasse davantage.
3. **Garder `auto`, remplacer le test d'aire par un test de recouvrement réel**
   (texels peints par deux étiquettes différentes). Attrape le cas des moitiés en
   miroir qui partagent la même zone de texture, garde le 1:1 sur les références.

### Faiblesses de l'heuristique actuelle

- Aveugle au recouvrement d'UV : des moitiés en miroir passent le test d'aire et
  le bake écrit deux couleurs au même endroit. Le bureau n'a été attrapé que parce
  que le recouvrement était extrême.
- Moyenne par pièce dominée par les grandes faces : real_c a 8 texels par face en
  moyenne mais 0,6 en médiane. Le seuil de 4 et le minimum de 50 faces sont deux
  chiffres magiques.
- Implémentation : le jonglage avec `has_uv` dans `bake_labels_to_glb` se réduit
  à un seul booléen « régénérer ».

## Autres points ouverts

- `complete_labels` : quelle étape crée les carrés sur le plateau de la table
  (raw correct, 909 faces) ; le 3-NN par `torch.cdist` alloue ≈ 16 Go sur 150k
  faces (cKDTree possible, attention aux égalités).
- Split real_c : pièces noires (0,0,0) issues de texels non peints ; tiroir vert.
- Remplacement des étapes de remplissage de `complete_labels` par le fill :
  expérience à mesurer, non décidée.
- `split_lab.py` : non suivi, à garder ou non.
- `main` : en avance de 6 / en retard de 1 sur origin, `--force-with-lease` à
  faire sur demande.
