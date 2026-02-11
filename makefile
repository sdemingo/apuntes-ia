# Declaramos los objetivos que no son archivos físicos
.PHONY: all draft doc clean

OUT_DIR = pdf

all: draft doc


PDF_OUT = pdf/draft.pdf

$(PDF_OUT): draft.md
	mkdir -p $(OUT_DIR)
	pandoc draft.md -o $(OUT_DIR)/draft.pdf --listings -Vlang=es-ES

draft: $(PDF_OUT)

doc: apuntes.tex
	mkdir -p $(OUT_DIR)
	latexmk -pdf -output-directory=$(OUT_DIR) apuntes.tex

clean:
	latexmk -c
