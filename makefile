all:
	mkdir -p pdf
	pandoc apuntes.md -o pdf/apuntes.pdf --listings -Vlang=es-ES

clean:
	rm -r pdf
