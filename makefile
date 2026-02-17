
# Configuración
OUT_DIR = pdf
PANDOC_FLAGS = --listings # -Vlang=es-ES

# Buscamos todos los archivos fuente
SOURCES_MD  := $(wildcard *.md)
SOURCES_TEX := $(wildcard *.tex)

# Definimos los nombres de los PDF resultantes
PDFS_MD  := $(patsubst %.md, $(OUT_DIR)/%.pdf, $(SOURCES_MD))
PDFS_TEX := $(patsubst %.tex, $(OUT_DIR)/%.pdf, $(SOURCES_TEX))

.PHONY: all clean directories

# Objetivo principal: compila todo
all: directories $(PDFS_MD) $(PDFS_TEX)

# Crea el directorio de salida si no existe
directories:
	@mkdir -p $(OUT_DIR)

# Regla de patrón para archivos Markdown -> PDF
$(OUT_DIR)/%.pdf: %.md
	pandoc $< -o $@ $(PANDOC_FLAGS)

# Regla de patrón para archivos LaTeX -> PDF
$(OUT_DIR)/%.pdf: %.tex
	latexmk -pdf -output-directory=$(OUT_DIR) $<

# Limpieza de archivos auxiliares y PDFs
clean:
	latexmk -c -output-directory=$(OUT_DIR)
	rm -rf $(OUT_DIR)
