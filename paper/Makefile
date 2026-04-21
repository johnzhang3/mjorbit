PDF := main.pdf
LATEXMK := latexmk

.PHONY: all clean

all: $(PDF)

$(PDF): main.tex references.bib sections/*.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -C main.tex
