all:
	mkdir -p pdf
	#pandoc draft.md -o pdf/draft.pdf --listings -Vlang=es-ES
	latexmk -pdf apuntes.tex
	mv apuntes.pdf pdf/

clean:
	latexmk -c

