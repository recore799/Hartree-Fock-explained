# Implementación del Método de Hartree-Fock (HF) en Python: Una Herramienta Educativa y de Investigación

Este proyecto busca proporcionar una implementación accesible y didáctica del método HF, útil para estudiantes e investigadores en química cuántica.

En presente se cuenta con un cálculo desde cero del método *Hartree-Fock restringido (RHF)* en Python, basada en el formalismo presentado en _Modern Quantum Chemistry_ de Szabo y Ostlund.
El código fue escrito con énfasis en claridad, modularidad y extensibilidad, con el objetivo de servir como base para métodos más avanzados (post-HF y DFT).

## Características

- Cálculo de integrales de un electrón
- Evaluación eficiente del tensor de 2-electrones `⟨μν|λσ⟩` con simetría
- Algoritmo SCF completo con ortogonalización simétrica o canónica
- Soporte para moléculas con base STO-3G y nucleos en 3D
- Resultados validados contra valores de referencia

## Resultados de ejemplo

A continuación se muestra un cálculo típico para las moléculas *H₂* y *HeH⁺* con base mínima STO-3G:

```text
=== SCF RESULTS FOR H₂ ===
------------------------------------------------------------
Number of basis functions: 2
Number of electrons: 2
Number of occupied orbitals: 1
Bond distance: 1.400 au

============================================================
FINAL SCF RESULTS
============================================================
Electronic energy:          -1.831001  Ha
Nuclear repulsion:           0.714286  Ha
Total energy:               -1.116715  Ha
SCF iterations:                     2
Converged:                       True

Orbital energies (Ha):
  ε_1 =  -1.107625
  ε_2 =   0.456238

Orbital Coefficients (C):
  Orbital 1 :   0.7071    0.7071
  Orbital 2 :   0.7071   -0.7071

=== SCF RESULTS FOR HeH⁺ ===
------------------------------------------------------------
Number of basis functions: 2
Number of electrons: 2
Number of occupied orbitals: 1
Bond distance: 1.463 au

============================================================
FINAL SCF RESULTS
============================================================
Electronic energy:          -4.227061  Ha
Nuclear repulsion:           1.366867  Ha
Total energy:               -2.860194  Ha
SCF iterations:                     7
Converged:                       True

Orbital energies (Ha):
  ε_1 =  -1.597080
  ε_2 =  -0.060864

Orbital Coefficients (C):
  Orbital 1 :  -0.8014   -0.3377
  Orbital 2 :   0.7823   -1.0678
```


### Como correr los cálculos
```text
python src/scf.py
```

## Extensiones

- Estoy trabajando en soporte para moléculas arbitrarias, en particular estoy tomando a la molécula LiH como una primera extensión. Para validar estos resultados estoy creando pruebas unitarias con referencia a los resultados dados por la libreria _pySCF_.

## Estructura del proyecto

```text
HF/
├── src/
│   ├── integrals.py         # Construcción de matrices S, T, V, eri
│   ├── scf.py               # Loop SCF y funciones auxiliares
│   └── sto3g-basis.py       # Construcción de la base en términos de primitivas gaussianas
├── tests/
│   ├── test_integrals.py    # Pruebas unitarias de matrices
│   └── ...
├── README.md
├── docs/
│   ├── HF.pdf                # Documentación del proceso SCF (en proceso)
│   └── HF.org                # Código fuente de la documentación
```


## Referencias

* Szabo, A., & Ostlund, N. S. (1996). Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory.


## Autor
Rafael Corella - a220211866@unison.mx
Estudiante de Física, Universidad de Sonora

Trabajo de investigación para presentación en congreso nacional de física 2025
