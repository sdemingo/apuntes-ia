
OUT_DIR = pdf
PANDOC_FLAGS = --listings # -Vlang=es-ES

SOURCES_MD  := $(wildcard *.md)
SOURCES_TEX := $(wildcard *.tex)

# Definimos los nombres de los PDF resultantes
PDFS_MD  := $(patsubst %.md, $(OUT_DIR)/%.pdf, $(SOURCES_MD))
PDFS_TEX := $(patsubst %.tex, $(OUT_DIR)/%.pdf, $(SOURCES_TEX))

.PHONY: all clean directories

all: directories $(PDFS_MD) $(PDFS_TEX)

directories:
	@mkdir -p $(OUT_DIR)

$(OUT_DIR)/%.pdf: %.md    # Markdown -> PDF
	pandoc $< -o $@ $(PANDOC_FLAGS)


$(OUT_DIR)/%.pdf: %.tex   # LaTeX -> PDF
	latexmk -pdf -output-directory=$(OUT_DIR) $<

clean:
	latexmk -c -output-directory=$(OUT_DIR)
	rm -rf $(OUT_DIR)
