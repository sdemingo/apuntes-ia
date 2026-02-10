all:
	mkdir -p pdf
	pandoc apuntes.md -o pdf/prueba.pdf --listings -Vlang=es-ES
