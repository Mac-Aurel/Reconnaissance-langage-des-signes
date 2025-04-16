# Langage des Signes - Reconnaissance par Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-red?logo=google)](https://google.github.io/mediapipe/)

Un projet de reconnaissance en **langue des signes** utilisant l'apprentissage profond (Deep Learning) et la détection en temps réel via webcam.

---

## 📁 Structure du projet

```
.
├── Langage_des_signes.ipynb     # Notebook d'entraînement du modèle
├── load_cam.py                  # Script de détection en temps réel
├── sign_language_model.h5       # (À générer) Modèle entraîné
```

---

## ⚙️ Installation

Clonez ce dépôt :

```bash
git clone https://github.com/votre-utilisateur/langage-des-signes.git
cd langage-des-signes
```

Installez les dépendances :

```bash
pip install -r requirements.txt
```

> 📌 Si `requirements.txt` n’est pas fourni, installez manuellement :

```bash
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python mediapipe tensorflow
```

---

## 🧠 Entraînement du modèle

Lancez le notebook :

```bash
jupyter notebook Langage_des_signes.ipynb
```

Ce notebook :
- Télécharge les données `sign-language-mnist`
- Entraîne un modèle CNN
- Sauvegarde le modèle dans `sign_language_model.h5`

---

## 🎥 Détection en temps réel

Une fois le modèle généré, lancez :

```bash
python load_cam.py
```

Le script :
- Ouvre la webcam
- Utilise MediaPipe pour détecter les mains
- Prédit la lettre signée à l’écran

---

## 🧪 Exemple de Résultat

| Geste | Prédiction Affichée |
|-------|----------------------|
| ✋ A   | `A`                  |
| 🤘 C   | `C`                  |

---

## 📌 Notes

- Précision du modèle dépendante de la position de la main et des conditions de lumière
- Reconnaissance uniquement de l’alphabet **A-Z**
- Potentielle amélioration : reconnaissance de mots complets, ajout de son, etc.

---

## 👨‍💻 Auteur

Projet réalisé par [**Oladé Laourou**](https://github.com/votre-utilisateur) — n'hésitez pas à contribuer ou poser vos questions via issues ou pull requests !

---

## 📄 Licence

Ce projet est sous licence MIT – voir le fichier [LICENSE](LICENSE) pour plus d’informations.
