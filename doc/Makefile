
DOT = dot
DOTOPTS = -Tps

TEX = latex

#-----------------------------
all: manual devel

manual: manual.tex
	$(TEX) manual.tex
	dvipdf manual.dvi

devel: devel.tex imogenInvoke.ps
	latex devel.tex
	dvipdf devel.dvi

imogenInvoke.ps: imogenInvoke.dot
	$(DOT) $(DOTOPTS) imogenInvoke.dot -o imogenInvoke.ps
