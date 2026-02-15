---
geometry: margin=2cm
numbersections: true
toc: true
title: "Apuntes sobre IA"
author: "Sergio de Mingo"
date: "2 de febrero de 2026"
header-includes: | 
  \lstset{
    frame=single,
    framesep=5pt,
    rulecolor=\color{black},
    basicstyle=\small\ttfamily,
    aboveskip=2em,   % Espacio antes del bloque
    belowskip=2em,    % Espacio después del bloque
    breaklines=true
  }
  \setlength{\parskip}{1em}
  \setlength{\parindent}{0pt}
---


\vspace{1cm}

# Redes Neuronales Convolucionales

Anteriormente aplanamos una imagen de 28×28 a 784. Al hacer eso, la red ya no
sabe si el píxel 1 está al lado del píxel 2 o del píxel 28. Simplemente ve una
lista de números. Piensa en esto: ¿Cómo podrías tú reconocer una cara si te
dieran todos los píxeles de una foto mezclados en una bolsa? Sería
imposible. Las CNN vienen a resolver precisamente eso.

